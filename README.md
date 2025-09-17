## 가상자산 조세포탈 대응 분석 툴 (PICA 거래 데이터)

### 개요
- 이 레포지토리는 PICA(ERC-20) 토큰 트랜잭션 데이터를 수집·변환하여 그래프 형태로 모델링하고, GraphSAGE 기반 오토인코더로 이상 거래(잠재적 조세포탈) 패턴을 탐지하는 파이프라인을 포함합니다.
- 파이프라인 구성: **T1 (CSV → SQLite, value 보정) → T2 (그래프 테이블 생성: addr/from_to/net_to) → T3 GraphSAGE 오토인코더 분석**

### 주요 파일
- `picaartmoney_transactions_full.csv` : 원본 트랜잭션 CSV (Etherscan 형식)
- `requirements.txt` : 필요한 Python 패키지 목록
- `src/t1_collect.py` : CSV 읽기 및 `value` 보정 후 SQLite로 저장 (T1)
- `src/t2_build_graph.py` : SQLite의 트랜잭션을 읽어 `addr`, `from_to`, `net_to` 생성 (T2)
- `src/t3_analyze.py` : Node2Vec + 단순 오토인코더 + XGBoost 분석 (참조용)
- `src/t3_graphsage.py` : GraphSAGE 기반 오토인코더 구현 (T3 주요 스크립트)
- `src/main.py` : 단계별 오케스트레이터 (t1/t2/t3/all)

### 설치
1. Python 3.8+ 환경을 준비합니다.
2. 종속성 설치:
```bash
pip install -r requirements.txt
```
- GPU(Pytorch) 사용을 원하면 OS/환경에 맞는 `torch` 바이너리를 설치하세요(공식 문서 권장).

### 실행 플로우 (권장 순서)
- 전체 실행(한 번에):
```bash
python -m src.main --stage all --csv picaartmoney_transactions_full.csv --db data/txs.db --out results/t3_graphsage.xlsx
```
- 단계별 실행:
  - T1 (CSV → SQLite, value 보정)
  ```bash
  python -m src.t1_collect --csv picaartmoney_transactions_full.csv --out data/txs.db
  ```
  - T2 (그래프 테이블 생성: `addr`, `from_to`, `net_to`)
  ```bash
  python -m src.t2_build_graph --db data/txs.db
  ```
  - T3 (GraphSAGE 오토인코더 분석)
  ```bash
  python -m src.t3_graphsage --db data/txs.db --out results/t3_graphsage.xlsx --epochs 200 --lr 0.001
  ```

### 출력물
- SQLite DB: `data/txs.db` (테이블: `transactions`, `addr`, `from_to`, `net_to`)
- 분석 결과 엑셀: `results/t3_graphsage.xlsx` (시트: `node_analysis`, `addr_table`, `node2vec_emb`, `xgb_importance`)

### GraphSAGE 오토인코더(방법 설명)
- 입력 특징:
  - Node2Vec 임베딩(64차원) + `addr` 테이블의 수치형 노드 속성 결합
- 인코더(합성):
  - GraphSAGE 계층을 2층 사용 (예: 256 -> 64)
  - 각 계층은 노드의 자기 특징과 이웃 평균 특징(concatenate)을 선형 변환 후 활성화(ReLU)
  - 이웃 샘플링: 구현은 간단한 무작위 샘플링(최대 k개) 방식(파이썬 루프)
- 디코더:
  - 임베딩(64d) → MLP → 입력 차원 재구성
- 학습 목표:
  - 입력 특징을 재구성하는 MSE 손실 최소화
  - 재구성 오차로 이상치 점수 계산(노드별 MSE → z-score → z>2 이상치)

### 하이퍼파라미터(권장값)
- Node2Vec: dimension=64, walks=200, walk_len=30, window=10
- GraphSAGE: hidden_dims=[256, 64], num_samples=10 (이웃 샘플링 수)
- 학습: epochs=200, lr=1e-3, weight_decay=1e-5
- 이상치 기준: recon_z > 2

### 성능·확장성 고려사항 (주의)
- 현재 구현은 전체 노드를 한 번에 불러와 학습하는 full-batch 방식이며, 이웃 샘플링은 순수 파이썬 루프입니다. 노드 수가 커지면 메모리 및 시간 병목이 발생합니다.
- 권장 확장 방향:
  - PyTorch Geometric(PyG) 또는 DGL로 포팅하여 효율적 미니배치 neighbor sampler 사용
  - GPU 학습: `torch` 설치 시 CUDA 빌드를 사용하고 모델/데이터를 `.cuda()`로 이동
  - 대규모 Node2Vec: gensim 파라미터(epochs, workers)와 워크 개수 조정 또는 병렬 워크 생성
  - SQLite I/O가 병목이면 Parquet/Feather 또는 DBMS(예: Postgres)로 전환

### 실습 팁
- 먼저 소규모 데이터(예: 1,000~5,000 노드)로 파라미터를 튜닝하세요.
- `--epochs` 및 `--lr`를 조정하면서 재구성 오차의 분포를 관찰하고 이상치 비율을 확인하세요.
