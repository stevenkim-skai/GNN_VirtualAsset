"""T2: 그래프 모델링 및 구축
- SQLite의 transactions 테이블을 읽어 addr(nodes), from_to(edges), net_to(요약 엣지)를 생성
- 간단한 노드 통계(총입/총출, 유니크 엣지 수)와 중심성(pagerank, betweenness)을 계산하여 addr 속성으로 저장
- 출력은 동일 SQLite DB에 새로운 테이블로 저장
"""
from pathlib import Path
import sqlite3
import pandas as pd
import networkx as nx


def build_graph(sqlite_path: str, tx_table: str = "transactions"):
    sqlite_path = Path(sqlite_path)
    conn = sqlite3.connect(str(sqlite_path))

    df = pd.read_sql(f"SELECT * FROM {tx_table}", conn)
    if df.empty:
        print("트랜잭션 테이블이 비어있습니다.")
        return

    # 표준 컬럼명
    fcol = "from" if "from" in df.columns else "sender"
    tcol = "to" if "to" in df.columns else "recipient"
    valcol = "value_adj" if "value_adj" in df.columns else "value"
    tscol = "timeStamp" if "timeStamp" in df.columns else "timestamp"

    # 노드 목록
    addrs = pd.Index(df[fcol].fillna("") .tolist() + df[tcol].fillna("").tolist()).unique()
    nodes_df = pd.DataFrame({"address": addrs})

    # 엣지 (원시)
    edges_df = df[[fcol, tcol, "hash", valcol, tscol]].copy()
    edges_df.columns = ["src", "dst", "txHash", "value", "timestamp"]
    edges_df["value"] = pd.to_numeric(edges_df["value"], errors="coerce").fillna(0.0)

    # NET_TO 집계
    net = (
        edges_df.groupby(["src", "dst"]).agg(
            netValue=("value", "sum"),
            netCount=("txHash", "count"),
            earliestTimestamp=("timestamp", "min"),
        )
        .reset_index()
    )

    # networkx 그래프 생성(요약 엣지 사용)
    G = nx.DiGraph()
    for _, r in net.iterrows():
        G.add_edge(r["src"], r["dst"], netValue=r["netValue"], netCount=int(r["netCount"]))

    # 노드 통계 계산
    pagerank = nx.pagerank(G, alpha=0.85) if len(G) > 0 else {}
    try:
        bet = nx.betweenness_centrality(G)
    except Exception:
        bet = {n: 0.0 for n in G.nodes()}

    # 집계 컬럼 삽입
    stats = []
    for n in nodes_df["address"]:
        in_edges = list(G.in_edges(n, data=True))
        out_edges = list(G.out_edges(n, data=True))
        totValIn = sum([e[2].get("netValue", 0.0) for e in in_edges])
        totValOut = sum([e[2].get("netValue", 0.0) for e in out_edges])
        totCntIn = sum([e[2].get("netCount", 0) for e in in_edges])
        totCntOut = sum([e[2].get("netCount", 0) for e in out_edges])
        uniqueEdgesIn = len(set([u for u,_,_ in in_edges]))
        uniqueEdgesOut = len(set([v for _,v,_ in out_edges]))
        stats.append(
            {
                "address": n,
                "totValIn": totValIn,
                "totValOut": totValOut,
                "totCntIn": totCntIn,
                "totCntOut": totCntOut,
                "uniqueEdgesIn": uniqueEdgesIn,
                "uniqueEdgesOut": uniqueEdgesOut,
                "centralityPagerank": float(pagerank.get(n, 0.0)),
                "centralityBetweenness": float(bet.get(n, 0.0)),
            }
        )

    stats_df = pd.DataFrame(stats)

    # SQLite에 저장
    stats_df.to_sql("addr", conn, if_exists="replace", index=False)
    edges_df.to_sql("from_to", conn, if_exists="replace", index=False)
    net.to_sql("net_to", conn, if_exists="replace", index=False)

    conn.close()
    print(f"Created addr ({len(stats_df)} rows), from_to ({len(edges_df)} rows), net_to ({len(net)} rows) in {sqlite_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="T2: SQLite 트랜잭션 -> 그래프 테이블 생성")
    parser.add_argument("--db", required=True, help="입력/출력 SQLite 파일 경로")
    parser.add_argument("--tx-table", default="transactions", help="트랜잭션 테이블 이름")
    args = parser.parse_args()
    build_graph(args.db, args.tx_table) 