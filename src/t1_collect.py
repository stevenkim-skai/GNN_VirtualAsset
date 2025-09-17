"""T1: 데이터 수집/변환 유틸리티
- CSV(etherscan 형식)를 읽어 SQLite DB에 저장
- value 컬럼을 tokenDecimal로 보정하여 실수 형태로 저장
- 기본 사용: python -m src.t1_collect --csv picaartmoney_transactions_full.csv --out data/txs.db
"""
from pathlib import Path
import argparse
import pandas as pd
import sqlite3


def normalize_value(value, token_decimal):
    try:
        iv = int(value)
    except Exception:
        try:
            iv = float(value)
        except Exception:
            return None
    try:
        td = int(token_decimal)
    except Exception:
        td = 0
    return float(iv) / (10 ** td)


def csv_to_sqlite(csv_path: str, sqlite_path: str, table_name: str = "transactions"):
    csv_path = Path(csv_path)
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 보정: value -> adjusted_value
    if "tokenDecimal" in df.columns and "value" in df.columns:
        df["value_adj"] = df.apply(
            lambda r: normalize_value(r["value"], r.get("tokenDecimal", 0)), axis=1
        )
    else:
        df["value_adj"] = pd.to_numeric(df["value"], errors="coerce")

    # 시간 컬럼 정리
    if "timeStamp" in df.columns:
        df["timeStamp"] = pd.to_numeric(df["timeStamp"], errors="coerce")

    # 저장
    conn = sqlite3.connect(str(sqlite_path))
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved {len(df)} rows to {sqlite_path}:{table_name}")


def main():
    parser = argparse.ArgumentParser(description="T1: CSV -> SQLite 변환 (value 보정)")
    parser.add_argument("--csv", required=True, help="입력 CSV 파일 경로")
    parser.add_argument("--out", required=True, help="출력 SQLite 파일 경로")
    parser.add_argument("--table", default="transactions", help="테이블 이름")
    args = parser.parse_args()
    csv_to_sqlite(args.csv, args.out, args.table)


if __name__ == "__main__":
    main() 