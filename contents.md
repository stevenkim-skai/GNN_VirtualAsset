# 발표자료용 콘텐츠 (코드 기반 프로젝트)

---

### 슬라이드 1 — 제목
- **프로젝트**: 그래프 기반 이상거래 탐지 (Node2Vec + GraphSAGE Autoencoder)
- **파일/스크립트**: `v2/graphsage.py`
- **목표**: 지갑별 재구성오차(Reconstruction Error)를 이용한 이상거래(상위 5%) 탐지 및 원인 분석

> 발표자 노트: 한 문장으로 문제와 해결책을 제시합니다. "블록체인 거래 그래프에서 GNN 기반 오토인코더로 이상 거래 패턴을 검출합니다."

---

### 슬라이드 2 — 코드 개요 (Code Overview)
- **핵심 파이프라인**:
  - 그래프 로딩 (NetworkX pickle, `node_features.csv`) → `edge_index` 생성
  - Node2Vec 임베딩 생성 (64차원)
  - GraphSAGE 기반 오토인코더 학습 (64 → 256 → 32 → 64)
  - 노드별 재구성 MSE 계산 → z-score 정규화 → 상위 5% 이상치 판정
  - 결과 저장: `sage_ae_anomalies.csv`, `sage_autoencoder.pt`, `node2vec_embeddings_sage.npy`
- **입력/출력**:
  - 입력: `node_features.csv`, `G_base_multidigraph.pkl` (네트워크 및 노드 속성)
  - 출력: 이상치 목록 CSV, 모델 체크포인트, 임베딩 numpy

> 발표자 노트: 코드의 입출력과 전체 흐름을 한 눈에 보여줍니다.

---

### 슬라이드 3 — 배경 (Background & Motivation)
- **문제**: 중앙화 거래소(CEX)와 직접 연결되지 않은 지갑의 익명성으로 인해 조세포탈·자금세탁 감지에 한계
- **접근**: 그래프 데이터베이스로 트랜잭션을 모델링하고, 비지도 학습(오토인코더)으로 이상거래 패턴 탐지
- **데이터셋 예시**: PICA (ERC-20) 트랜잭션 (논문 `paper.md` 참고)

> 발표자 노트: 왜 그래프 모델과 오토인코더를 선택했는지 짧게 설명합니다.

---

### 슬라이드 4 — 아키텍처 및 코드 구조 (Architecture / Structure)
- **모듈/스크립트**:
  - `v2/graphsage.py`: 전체 파이프라인(임베딩 → AE 학습 → 이상치 검출)
  - 데이터 파일: `picaartmoney_transactions_full.csv`, `G_base_multidigraph.pkl`, `node_features.csv`
  - 문서: `paper.md` (방법론·파라미터·해석 근거)
- **논리 구조 (단계)**:
  1. 데이터 로딩 및 인덱싱
  2. Node2Vec 학습 (PyG)
  3. GraphSAGE 오토인코더 정의 및 학습
  4. 재구성오차 계산 → z-score → 이상치 결정
  5. 결과 저장 및 후속 분석(XGBoost/SHAP 등)

> 발표자 노트: 파일 맵과 코드 내 핵심 함수/클래스(`SAGEAutoEncoder`, Node2Vec 객체 등)를 짚어줍니다.

---

### 슬라이드 5 — 과정별 상세 (1) 데이터·모델 준비
- **Skills**: 데이터 전처리, 그래프 변환, 인덱싱
- **Platform**: 로컬/서버 (CUDA 권장), Docker 가능
- **Tech / Architect**: NetworkX → PyG `Data(edge_index, num_nodes)`
- **Libraries / Frameworks**: `pandas`, `networkx`, `torch`, `torch_geometric`
- **Key points**:
  - 노드 매핑(`addr2idx`)으로 `edge_index` 생성
  - 멀티다이그래프를 단일 edge list로 변환(필요시 양방향 추가)

> 발표자 노트: 입력 데이터의 품질(노드 속성 누락, 희소성)에 따른 영향 언급.

---

### 슬라이드 6 — 과정별 상세 (2) Node2Vec 임베딩
- **Skills**: 무작위 워크 설정, 하이퍼파라미터 튜닝
- **Platform**: GPU 권장 (PyG의 Node2Vec는 SparseAdam 사용 가능)
- **Tech**: 랜덤 워크 기반 임베딩(구조+속성 반영)
- **Libraries**: `torch_geometric.nn.Node2Vec`
- **Typical params (코드 기반)**:
  - embedding_dim=64, walk_length=30, context_size=10, walks_per_node=200, num_negative_samples=1
  - optimizer: `SparseAdam`, lr=0.01
- **Key points**:
  - 충분한 walks와 negative samples로 안정적 임베딩 확보
  - 임베딩 저장 후 AE 입력으로 사용

> 발표자 노트: 임베딩 품질이 이상 탐지 민감도에 큰 영향이 있음을 강조.

---

### 슬라이드 7 — 과정별 상세 (3) GraphSAGE 오토인코더
- **Skills**: GNN 모델 설계, 레이어·차원 결정, 정규화 전략
- **Architecture**: `SAGEConv` 3계층 (in 64 → 256 → 32 → 64 out)
- **Activation / Regularization**: ELU, Dropout=0.3
- **Loss / Optimizer**: MSE loss, `Adam` (lr=1e-3, weight_decay=5e-4)
- **Training**:
  - 입력: Node2Vec 임베딩
  - 목표: 입력 임베딩을 재구성하여 노드별 MSE 최소화
  - Epochs: 예시 1000 (코드), 모니터링/early stopping 권장
- **Key points**:
  - 그래프 구조를 반영한 메시지 패싱이 재구성 성능에 영향
  - overfitting 방지를 위한 dropout 및 weight decay

> 발표자 노트: 인코더-디코더가 아닌, SAGEConv를 통해 직접 복원하는 AE 형태임을 설명.

---

### 슬라이드 8 — 과정별 상세 (4) 이상치 판정 및 평가
- **재구성오차 계산**: 노드별 평균 MSE (feature 차원 기준)
- **정규화**: z-score (표본 표준편차 사용, sigma=std(ddof=1))
- **임계값**: 상위 5% (z percentile 95%) → `is_anomaly_top5pct`
- **평가/해석**:
  - 결과를 `sage_ae_anomalies.csv`로 저장
  - XGBoost로 재구성오차의 주요 원인(피처 중요도, SHAP) 분석 권장

> 발표자 노트: 임계값 선택(95%)은 사례·도메인에 따라 조정 가능함을 명시.

---

### 슬라이드 9 — 과정별 상세 (5) 운영·배포·확장 고려사항
- **Reproducibility**: 시드 고정(`torch.manual_seed`, `np.random.seed`), 모델·임베딩 저장
- **Scaling**:
  - 큰 그래프: 워크셋 샘플링, 배치 임베딩, 분산 학습 고려
  - PyG의 메모리 한계: sparse 연산, mini-batch 전략
- **Deployment**: 모델 체크포인트와 CSV 결과를 파이프라인에 연동(알림·대시보드)
- **환경**: CUDA-enabled GPU 권장, Dockerfile 및 requirements 관리

> 발표자 노트: 실무 적용 시 데이터 파이프라인(정기 수집→Graph 구축→분석)을 자동화할 것을 권장.

---

### 슬라이드 10 — 사용된 주요 라이브러리·툴 요약
- **분석·프레임워크**: `torch`, `torch_geometric`, `pandas`, `networkx`, `numpy`
- **임베딩**: `torch_geometric.nn.Node2Vec`
- **GNN 레이어**: `torch_geometric.nn.SAGEConv`
- **모델 저장/결과**: `torch.save`, `numpy.save`, CSV
- **추가 분석**: `xgboost`, `shap`, Excel(.xlsx) 리포트

> 발표자 노트: 각 라이브러리의 역할을 한 줄로 정리해서 제시.

---

### 슬라이드 11 — 발표용 핵심 메시지 (결론)
- **핵심**: 그래프 임베딩 + GraphSAGE 오토인코더는 라벨 없는 블록체인 데이터에서도 유효한 이상거래 탐지 기법을 제공한다.
- **실용성**: 결과를 XGBoost/SHAP로 해석하면 조사 우선순위(조세포탈 의심 지갑)를 도출할 수 있음
- **권장**: 임계값·하이퍼파라미터 튜닝, 대규모 데이터셋 적용을 위한 인프라 보완

---

### 부록 — 발표자 메모: 데모 시나리오
- 로컬에서 데모 실행 순서:
  1. `node_features.csv`, `G_base_multidigraph.pkl` 준비
  2. `python v2/graphsage.py` 실행 (CUDA 환경 권장)
  3. 생성된 `sage_ae_anomalies.csv` 확인 → 상위 지갑 사례 2~3개 시연
  4. XGBoost 분석 결과로 주요 원인·SHAP 예시 시연

---

### 부록 2 — 슬라이드별 코드 (발표용)
- **설명**: 각 슬라이드에서 바로 보여주기 좋은 최소한의 코드 스니펫입니다. 전체 코드는 `v2/graphsage.py`를 참고하세요.

#### 슬라이드 2 — 코드 개요 (파이프라인 요약)
```python
# main pipeline (요약)
# 1) 데이터 로드
# 2) Node2Vec 학습
# 3) GraphSAGE AE 학습
# 4) 이상치 판정 및 저장

# (간단 의사코드)
G = load_graph_from_pickle('G_base_multidigraph.pkl')
node_features = pd.read_csv('node_features.csv')
edge_index, addresses = build_edge_index(G, node_features)

n2v = train_node2vec(edge_index)
x_init = extract_node2vec_embeddings(n2v)

model = SAGEAutoEncoder(in_dim=64)
train_sage_ae(model, x_init, edge_index)

anomalies = detect_anomalies(model, x_init, edge_index)
save_results(anomalies, 'sage_ae_anomalies.csv')
```

#### 슬라이드 5 — 데이터·모델 준비 (그래프 로딩 및 edge_index 생성)
```python
# Graph 로딩 및 edge_index 생성
import pickle
with open('G_base_multidigraph.pkl', 'rb') as f:
    G_nx = pickle.load(f)  # MultiDiGraph

addresses = node_df['address'].astype(str).tolist()
addr2idx = {a: i for i, a in enumerate(addresses)}

edges = []
for u, v, _k in G_nx.edges(keys=True):
    if u in addr2idx and v in addr2idx:
        edges.append([addr2idx[u], addr2idx[v]])
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

data = Data(edge_index=edge_index, num_nodes=len(addresses)).to(DEVICE)
```

#### 슬라이드 6 — Node2Vec 임베딩
```python
# Node2Vec (PyG) 요약
n2v = Node2Vec(
    data.edge_index,
    embedding_dim=64,
    walk_length=30,
    context_size=10,
    walks_per_node=200,
    p=1.0, q=1.0,
    num_negative_samples=1,
    sparse=True
).to(DEVICE)

loader = n2v.loader(batch_size=128, shuffle=True, num_workers=4)
opt = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)

n2v.train()
for pos_rw, neg_rw in loader:
    opt.zero_grad()
    loss = n2v.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
    loss.backward()
    opt.step()

with torch.no_grad():
    x_init = n2v.embedding.weight.clone().detach()  # (N, 64)
```

#### 슬라이드 7 — GraphSAGE 오토인코더 (모델 정의)
```python
# GraphSAGE AutoEncoder (요약)
class SAGEAutoEncoder(nn.Module):
    def __init__(self, in_dim=64, dropout=0.3, aggr='mean'):
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
        x = self.sage3(x, edge_index)
        return x
```

#### 슬라이드 8 — 이상치 판정 (계산 및 저장)
```python
# 재구성오차 계산 및 이상치 판정
model.eval()
with torch.no_grad():
    x_recon = model(x_in, data.edge_index)

recon_err = torch.mean((x_recon - x_in)**2, dim=1).detach().cpu().numpy()
mu = recon_err.mean()
sigma = recon_err.std(ddof=1) if recon_err.size > 1 else 1e-8
z = (recon_err - mu) / (sigma if sigma > 0 else 1e-8)

z_cut = np.percentile(z, 95.0)
anom = (z > z_cut).astype(int)

out = pd.DataFrame({
    'address': addresses,
    'recon_mse': recon_err,
    'z_score': z,
    'is_anomaly_top5pct': anom
})
out.sort_values('z_score', ascending=False).to_csv('sage_ae_anomalies.csv', index=False)
```

--- 