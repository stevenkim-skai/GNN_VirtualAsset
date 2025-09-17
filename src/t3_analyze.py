"""T3: AI 기반 그래프 분석
- net_to/addr 테이블을 읽어 Node2Vec 임베딩(64d)을 생성
- 간단한 오토인코더(MLP 기반)로 재구성 오차를 계산하여 이상치 선별
- XGBoost로 주요 영향 변수(중요도)를 계산
- 결과를 엑셀로 저장
"""
from pathlib import Path
import sqlite3
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb


def build_walks(G: nx.Graph, num_walks=200, walk_length=30):
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for n in nodes:
            walk = [n]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.successors(cur)) if G.is_directed() else list(G.neighbors(cur))
                if neighbors:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append([str(x) for x in walk])
    return walks


def node2vec_embeddings(net_df: pd.DataFrame, dim=64, walks=200, walk_len=30):
    # 그래프 생성
    G = nx.DiGraph()
    for _, r in net_df.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]))

    walks = build_walks(G, num_walks=walks, walk_length=walk_len)
    model = Word2Vec(walks, vector_size=dim, window=10, min_count=1, sg=1, workers=2, epochs=4)

    emb = {n: model.wv[str(n)] for n in G.nodes()}
    emb_df = pd.DataFrame.from_dict(emb, orient="index")
    emb_df.index.name = "address"
    emb_df.reset_index(inplace=True)
    return emb_df


def autoencoder_reconstruction(emb_df: pd.DataFrame):
    feature_cols = [c for c in emb_df.columns if c != "address"]
    X = emb_df[feature_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 간단한 MLP 오토인코더: 입력->256->32->256->입력
    # sklearn MLPRegressor로 입력을 재구성하도록 학습
    X_train, X_val = train_test_split(Xs, test_size=0.2, random_state=42)
    ae = MLPRegressor(hidden_layer_sizes=(256, 32, 256), activation="relu", max_iter=200, random_state=42)
    ae.fit(X_train, X_train)
    rec = ae.predict(Xs)
    mse = np.mean((Xs - rec) ** 2, axis=1)
    # z-score
    z = (mse - mse.mean()) / (mse.std(ddof=0) + 1e-12)
    res = emb_df[["address"]].copy()
    res["recon_mse"] = mse
    res["recon_z"] = z
    res["anomaly_yn"] = (z > 2).astype(int)
    return res


def xgboost_analysis(nodes_df: pd.DataFrame, anomaly_series: pd.Series):
    # nodes_df: addr 테이블의 속성들
    df = nodes_df.copy()
    df = df.set_index("address")
    y = anomaly_series.reindex(df.index).fillna(0).astype(int)
    # 숫자형 피처만 사용
    X = df.select_dtypes(include=["number"]).fillna(0)
    if X.shape[0] < 10:
        print("데이터가 작아 XGBoost 학습을 건너뜁니다.")
        return None
    Xs = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    importance = model.get_booster().get_score(importance_type="weight")
    imp_df = pd.DataFrame([{"feature": k, "importance": v} for k, v in importance.items()])
    return model, imp_df.sort_values("importance", ascending=False)


def export_excel(out_path: str, node_results: pd.DataFrame, nodes_df: pd.DataFrame, emb_df: pd.DataFrame, imp_df=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        node_results.to_excel(w, sheet_name="node_analysis", index=False)
        nodes_df.to_excel(w, sheet_name="addr_table", index=False)
        emb_df.to_excel(w, sheet_name="node2vec_emb", index=False)
        if imp_df is not None:
            imp_df.to_excel(w, sheet_name="xgb_importance", index=False)
    print(f"Saved analysis to {out_path}")


def run_t3(sqlite_path: str, out_excel: str = "results/t3_results.xlsx"):
    conn = sqlite3.connect(sqlite_path)
    net = pd.read_sql("SELECT src, dst, netValue, netCount, earliestTimestamp FROM net_to", conn)
    nodes = pd.read_sql("SELECT * FROM addr", conn)
    conn.close()

    emb_df = node2vec_embeddings(net, dim=64, walks=200, walk_len=30)
    node_res = autoencoder_reconstruction(emb_df)

    # 노드 속성 인덱스 정렬
    nodes_indexed = nodes.set_index("address").reindex(node_res["address"]).reset_index()

    xgb_model, imp_df = xgboost_analysis(nodes_indexed, node_res.set_index("address")["anomaly_yn"]) or (None, None)

    export_excel(out_excel, node_res, nodes_indexed, emb_df, imp_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="T3: Node2Vec + simple autoencoder + XGBoost 분석")
    parser.add_argument("--db", required=True, help="입력 SQLite 파일 경로")
    parser.add_argument("--out", default="results/t3_results.xlsx", help="출력 엑셀 파일")
    args = parser.parse_args()
    run_t3(args.db, args.out) 