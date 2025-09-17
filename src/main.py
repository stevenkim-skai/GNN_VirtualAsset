"""오케스트레이터 스크립트
- 단계: t1, t2, t3 또는 all
- 예: python src/main.py --stage all --csv picaartmoney_transactions_full.csv --db data/txs.db
"""
import argparse
from pathlib import Path
import subprocess
import sys


def run_stage_t1(csv_path, db_path):
    print("Running T1: CSV -> SQLite")
    subprocess.check_call([sys.executable, "-m", "src.t1_collect", "--csv", csv_path, "--out", db_path])


def run_stage_t2(db_path):
    print("Running T2: Build graph tables")
    subprocess.check_call([sys.executable, "-m", "src.t2_build_graph", "--db", db_path])


def run_stage_t3(db_path, out_xlsx):
    print("Running T3: Analysis")
    subprocess.check_call([sys.executable, "-m", "src.t3_analyze", "--db", db_path, "--out", out_xlsx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["t1", "t2", "t3", "all"], default="all")
    parser.add_argument("--csv", default="picaartmoney_transactions_full.csv")
    parser.add_argument("--db", default="data/txs.db")
    parser.add_argument("--out", default="results/t3_results.xlsx")
    args = parser.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    if args.stage in ("t1", "all"):
        run_stage_t1(args.csv, args.db)
    if args.stage in ("t2", "all"):
        run_stage_t2(args.db)
    if args.stage in ("t3", "all"):
        run_stage_t3(args.db, args.out)


if __name__ == "__main__":
    main() 