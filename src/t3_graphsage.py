"""T3 GraphSAGE 기반 오토인코더 분석
- Node2Vec 임베딩과 노드 속성을 결합한 입력 특징으로 GraphSAGE 인코더를 학습
- 디코더(MLP)로 입력 특징을 재구성하여 재구성 오차 기반 이상치 탐지
- XGBoost 분석 및 엑셀 결과 저장은 기존 유틸리티를 재사용
"""
from pathlib import Path
import sqlite3
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import numpy as np
from src.t3_analyze import node2vec_embeddings, xgboost_analysis, export_excel


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, h, neigh_mean):
        # h: (N, in_dim), neigh_mean: (N, in_dim)
        cat = torch.cat([h, neigh_mean], dim=1)
        return torch.relu(self.linear(cat))


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(GraphSAGELayer(dims[i], dims[i + 1]))

    def forward(self, h, adj, node_list, node_to_idx, num_samples=10):
        # full-batch sampling: for each layer, compute neighbor mean for each node
        for layer in self.layers:
            # compute neighbor mean
            neigh_mean = torch.zeros_like(h)
            for i, node in enumerate(node_list):
                neigh = adj.get(node, [])
                if len(neigh) == 0:
                    continue
                # sample neighbors
                if len(neigh) > num_samples:
                    sampled = np.random.choice(neigh, num_samples, replace=False)
                else:
                    sampled = neigh
                idxs = [node_to_idx[n] for n in sampled if n in node_to_idx]
                if len(idxs) == 0:
                    continue
                neigh_mean[i] = h[idxs].mean(dim=0)
            h = layer(h, neigh_mean)
        return h


class Decoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, z):
        return self.net(z)


def build_adj_from_net(net_df: pd.DataFrame):
    adj = {}
    for _, r in net_df.iterrows():
        s = str(r["src"])
        d = str(r["dst"])
        adj.setdefault(s, []).append(d)
        # include reverse for undirected neighbor information (optional)
        adj.setdefault(d, [])
    return adj


def prepare_features(net_df: pd.DataFrame, nodes_df: pd.DataFrame):
    # Node2Vec 임베딩
    emb_df = node2vec_embeddings(net_df, dim=64, walks=200, walk_len=30)

    emb_df = emb_df.rename(columns={0: "e0"}) if 0 in emb_df.columns else emb_df
    emb_df = emb_df.set_index("address")

    # numeric features from nodes_df
    num_cols = nodes_df.select_dtypes(include=["number"]).columns.tolist()
    nodes_numeric = nodes_df.set_index("address")[num_cols].fillna(0)

    # join (reindex to emb nodes)
    common = [n for n in emb_df.index if n in nodes_numeric.index]
    emb_sub = emb_df.loc[common]
    num_sub = nodes_numeric.loc[common]

    X_emb = np.vstack(emb_sub.drop(columns=[c for c in emb_sub.columns if c == "address"], errors="ignore")) if not emb_sub.empty else np.zeros((0, 64))
    X_num = num_sub.values if not num_sub.empty else np.zeros((0, len(num_cols)))

    X = np.hstack([X_emb, X_num]) if X_num.shape[1] > 0 else X_emb

    feature_df = pd.DataFrame(X, index=common)
    feature_df.index.name = "address"
    feature_df.reset_index(inplace=True)
    return feature_df, emb_sub.reset_index(), num_cols


def run_graphsage(sqlite_path: str, out_excel: str = "results/t3_graphsage.xlsx", epochs: int = 200, lr: float = 1e-3):
    conn = sqlite3.connect(sqlite_path)
    net = pd.read_sql("SELECT src, dst, netValue, netCount, earliestTimestamp FROM net_to", conn)
    nodes = pd.read_sql("SELECT * FROM addr", conn)
    conn.close()

    feature_df, emb_df, num_cols = prepare_features(net, nodes)
    if feature_df.empty:
        print("features empty — GraphSAGE 실행 불가")
        return

    node_list = feature_df["address"].astype(str).tolist()
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    adj = build_adj_from_net(net)

    X = feature_df.drop(columns=["address"]).values.astype(float)
    X = torch.tensor(X, dtype=torch.float32)

    input_dim = X.shape[1]
    encoder = GraphSAGEEncoder(input_dim, [256, 64])
    decoder = Decoder(64, 256)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # training loop (full-batch)
    for ep in range(epochs):
        encoder.train()
        decoder.train()
        # initial features as h0
        h = X.clone()
        z = encoder(h, adj, node_list, node_to_idx, num_samples=10)
        rec = decoder(z)
        loss = loss_fn(rec, X)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} loss={loss.item():.6f}")

    # evaluation
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = encoder(X, adj, node_list, node_to_idx, num_samples=10)
        rec = decoder(z)
        mse = torch.mean((X - rec) ** 2, dim=1).cpu().numpy()

    # z-score
    zscore = (mse - mse.mean()) / (mse.std(ddof=0) + 1e-12)
    node_results = pd.DataFrame({"address": node_list, "recon_mse": mse, "recon_z": zscore})
    node_results["anomaly_yn"] = (node_results["recon_z"] > 2).astype(int)

    # nodes table aligned
    nodes_indexed = nodes.set_index("address").reindex(node_results["address"]).reset_index()

    # XGBoost analysis
    xgb_model, imp_df = xgboost_analysis(nodes_indexed, node_results.set_index("address")["anomaly_yn"]) or (None, None)

    export_excel(out_excel, node_results, nodes_indexed, emb_df.reset_index(), imp_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="T3 GraphSAGE autoencoder 분석")
    parser.add_argument("--db", required=True, help="입력 SQLite 파일 경로")
    parser.add_argument("--out", default="results/t3_graphsage.xlsx", help="출력 엑셀 파일")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    run_graphsage(args.db, args.out, epochs=args.epochs, lr=args.lr) 