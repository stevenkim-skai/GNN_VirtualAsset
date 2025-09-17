# -*- coding: utf-8 -*-
"""
Node2Vec(64d) + GraphSAGE 오토인코더(256->32->64) 학습 후
재구성오차 z-score 정규화, 상위 5% 이상치 탐지
필요: torch, torch_geometric, pandas, networkx, numpy
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec, SAGEConv
from torch_geometric.data import Data

# ------------------------------------------------------------
# 경로/환경 설정
NODE_FEAT_CSV = "node_features.csv"
GRAPH_PKL = "G_base_multidigraph.pkl"   # 이전 단계 저장 파일
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------
# 0) 노드/그래프 로딩
node_df = pd.read_csv(NODE_FEAT_CSV)
addresses = node_df['address'].astype(str).tolist()
addr2idx = {a: i for i, a in enumerate(addresses)}

if not os.path.exists(GRAPH_PKL):
    raise FileNotFoundError(f"{GRAPH_PKL} 파일이 없습니다. 경로를 확인하세요.")

with open(GRAPH_PKL, "rb") as f:
    G_nx = pickle.load(f)  # MultiDiGraph

# edge_index (방향 그래프). 필요 시 무방향으로 만들려면 아래 주석 해제
edges = []
for u, v, _k in G_nx.edges(keys=True):
    if u in addr2idx and v in addr2idx:
        edges.append([addr2idx[u], addr2idx[v]])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# # (선택) 양방향 추가: GNN 안정화에 도움이 될 수 있음
# edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

num_nodes = len(addresses)
data = Data(edge_index=edge_index, num_nodes=num_nodes).to(DEVICE)

print(f"장치: {DEVICE}, 노드 수: {num_nodes}, 엣지 수: {edge_index.size(1)}")

# ------------------------------------------------------------
# 1) Node2Vec 임베딩(64차원) 생성
# 요청 파라미터 매핑:
#   embedding_dim=64, walk_length=30, context_size=10(window),
#   walks_per_node=200(num_walk), p=1, q=1, workers=4
# (min_count, batch_words는 gensim 전용 → PyG에는 해당 없음)
n2v = Node2Vec(
    data.edge_index,
    embedding_dim=64,
    walk_length=30,
    context_size=10,
    walks_per_node=200,
    p=1.0,
    q=1.0,
    num_negative_samples=1,
    sparse=True
).to(DEVICE)

n2v_loader = n2v.loader(batch_size=128, shuffle=True, num_workers=4)
n2v_optimizer = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)

def train_node2vec(epochs=5):
    n2v.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for pos_rw, neg_rw in n2v_loader:
            n2v_optimizer.zero_grad()
            loss = n2v.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
            loss.backward()
            n2v_optimizer.step()
            total_loss += loss.item()
        print(f"[Node2Vec] epoch {epoch:03d} | loss {total_loss/len(n2v_loader):.4f}")

train_node2vec(epochs=5)

with torch.no_grad():
    x_init = n2v.embedding.weight.clone().detach()  # (N, 64)

# ------------------------------------------------------------
# 2) GraphSAGE 오토인코더
# - 64 -> 256 -> 32 -> 64
# - 활성화 ELU, 드롭아웃 0.3
# - 손실 MSE (입력 x_init 대비 복원 x_hat)

class SAGEAutoEncoder(nn.Module):
    def __init__(self, in_dim=64, dropout=0.3, aggr="mean"):
        super().__init__()
        self.dropout = dropout
        self.sage1 = SAGEConv(in_channels=in_dim, out_channels=256, aggr=aggr)
        self.sage2 = SAGEConv(in_channels=256, out_channels=32, aggr=aggr)
        self.sage3 = SAGEConv(in_channels=32, out_channels=64, aggr=aggr)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage3(x, edge_index)  # 복원 64d (보통 마지막은 활성 미적용)
        return x

model = SAGEAutoEncoder(in_dim=64, dropout=0.3, aggr="mean").to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

# ------------------------------------------------------------
# 3) 학습 (1000 epoch, MSE 재구성오차 최소화)
x_in = x_init.to(DEVICE)

def train_sage_ae(epochs=1000):
    model.train()
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        x_hat = model(x_in, data.edge_index)
        loss = F.mse_loss(x_hat, x_in)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"[SAGE-AE] epoch {epoch:04d} | recon MSE {loss.item():.6f}")

train_sage_ae(epochs=1000)

# ------------------------------------------------------------
# 4) 재구성오차 → z-score → 상위 5% 이상치
model.eval()
with torch.no_grad():
    x_recon = model(x_in, data.edge_index)

recon_err = torch.mean((x_recon - x_in)**2, dim=1).detach().cpu().numpy()
mu = recon_err.mean()
sigma = recon_err.std(ddof=1) if recon_err.size > 1 else 1e-8
z = (recon_err - mu) / (sigma if sigma > 0 else 1e-8)

z_cut = np.percentile(z, 95.0)
anom = (z > z_cut).astype(int)

print(f"상위 5% z-score 임계값: {z_cut:.4f}")
print(f"이상치 노드 수: {anom.sum()} / {len(anom)}")

# ------------------------------------------------------------
# 5) 결과 저장
out = pd.DataFrame({
    "address": addresses,
    "recon_mse": recon_err,
    "z_score": z,
    "is_anomaly_top5pct": anom
})
out.sort_values("z_score", ascending=False).to_csv("sage_ae_anomalies.csv", index=False, encoding="utf-8")

torch.save(model.state_dict(), "sage_autoencoder.pt")
np.save("node2vec_embeddings_sage.npy", x_init.detach().cpu().numpy())

print("✅ 완료: sage_ae_anomalies.csv / sage_autoencoder.pt / node2vec_embeddings_sage.npy 생성")
