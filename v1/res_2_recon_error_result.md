## 재구성 오차( Reconstruction Error ) 분석 결과

`recon_errors_summary.csv` 요약

| 모델 | val_count | mean (MSE) | std (MSE) | anomalies (z>2) | anomalies (z>3) | csv 파일 |
|---|---:|---:|---:|---:|---:|---|
| GCN_AE | 1592 | 1,778,398.375 | 16,291,265.0 | 1 | 1 | `recon_errors_GCN_AE.csv` |
| SAGE_AE | 1592 | 59,957.285 | 40,967.734 | 156 | 4 | `recon_errors_SAGE_AE.csv` |
| GAT_AE | 1592 | 3,018,049.25 | 1,117,918.625 | 0 | 0 | `recon_errors_GAT_AE.csv` |
| PaperGAT_AE | 1592 | 4,197,445.5 | 2,067,450.25 | 0 | 0 | `recon_errors_PaperGAT_AE.csv` |

핵심 발견
- **SAGE_AE가 z>2 기준으로 가장 많은 이상치(156개, 약 9.8%)를 탐지**했습니다(이 중 z>3는 4개).
- **GCN_AE는 z>2, z>3 각각 1건씩**으로 소수의 극단값이 존재합니다.
- **GAT_AE, PaperGAT_AE는 현재 z>2/3 기준에서 이상치가 탐지되지 않았습니다.**

z > 2인 이상치(anomalies)
Z-점수가 2보다 크면 데이터 포인트가 평균에서 2 표준편차 이상 떨어져 있음을 의미합니다. 정규 분포(종 모양 곡선)에서 평균의 ±2 표준편차 범위 밖의 데이터는 전체 데이터의 약 **4.55%**에 해당합니다. 따라서 이 기준은 상대적으로 덜 엄격하며 잠재적인 이상치를 식별하는 데 사용될 수 있습니다.

z > 3인 이상치(anomalies)
Z-점수가 3보다 크면 데이터 포인트가 평균에서 3 표준편차 이상 떨어져 있음을 의미합니다. 정규 분포에서 평균의 ±3 표준편차 범위 밖의 데이터는 전체 데이터의 약 **0.27%**에 불과합니다. 이 기준은 더 엄격한 이상치 탐지 방법이며, 매우 드물거나 극단적인 데이터 포인트를 식별하는 데 사용됩니다.

해석
- SAGE_AE의 평균 재구성 오차는 다른 모델보다 훨씬 작지만(약 6e4), 표준편차 대비 일부 노드에서 상대적으로 큰 오차를 보여 다수의 z>2 이상치가 발생했습니다. 즉, SAGE는 전반적 재구성 성능은 좋으나 특정 노드에서 재구성이 크게 실패합니다.
- GCN_AE의 평균(≈1.8e6)에 비해 표준편차가 매우 큽니다(≈1.6e7). 이는 소수의 극단적인 오차가 전체 분포에 강한 영향을 준 결과로 보이며, 해당 극단값 자체가 이상치(또는 계산/스케일 문제)일 가능성이 있습니다.
- GAT 계열(PaperGAT 포함)은 평균 오차가 크지만 표준편차가 상대적으로 작아 z 기준으로는 이상치가 나오지 않았습니다(분포가 넓게 퍼져 있는 모양).

권장 조치
- 우선 `SAGE_AE`의 이상치 목록(`recon_errors_SAGE_AE.csv`)을 열어 상위(z>3 또는 재구성 오차 상위) 주소를 확인하세요. 거래소 태그(`is_exchange`, `label` 등)가 노드 속성에 있다면 교차검증을 수행합니다.

간단 확인(파이썬)
```python
import pandas as pd
# SAGE 상위 이상치 확인
df = pd.read_csv('recon_errors_SAGE_AE.csv')
print('총 검증 수:', len(df))
print('z>3 개수:', df['anomaly_z3'].sum())
print(df.sort_values('zscore', ascending=False).head(20))

# 분포 확인
import matplotlib.pyplot as plt
plt.hist(df['recon_error'], bins=100)
plt.yscale('log')
plt.title('Reconstruction error distribution (SAGE_AE)')
plt.show()
```

- **피처 스케일링 확인**: 입력 피처 스케일(특히 `uniqedgesIn` 등)이 클 경우 `log1p` 또는 `StandardScaler` 적용 후 재학습하여 이상치 분포 변화 확인.
- **상위 이상치 수동 조사**: z>3로 탐지된 주소를 블록/트랜잭션 원본에서 조회하여 실제 의심거래(거래소 입출금, 급격한 송금 등)와 연관되는지 확인.
- **임계값 튜닝**: z>2/3 외에 상위 k% (예: 상위 1% 노드) 기준도 병행 적용하여 안정성 검토.
- **시각화**: 재구성 오차의 t-SNE/UMAP(또는 재구성 오차를 색상으로)으로 노드 분포를 시각화하면 이상치 패턴 파악에 도움이 됩니다.
