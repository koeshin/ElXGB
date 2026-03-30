# ELXGB Training Run Report

## Summary

ELXGB 학습 알고리즘을 실제 데이터셋에 대해 직접 실행했고, 학습 시간과 communication cost를 수집했습니다.

- Execution date: 2026-03-30
- Branch: `jina`
- Dataset: `sklearn.datasets.load_breast_cancer()`
- Goal: ELXGB 학습이 실제로 끝까지 동작하는지 확인하고, 시간/통신량을 기록

## Run Configuration

| Item | Value |
| --- | --- |
| Train samples | 398 |
| Test samples | 171 |
| Features | 30 |
| Number of parties | 2 |
| Feature split | 15 / 15 |
| Trees | 2 |
| Max depth | 2 |
| Bins | 32 |
| Epsilon (`eps`) | 0.03125 |
| Learning rate | 1.0 |
| DP epsilon | 10.0 |
| Random state | 42 |

## Execution Method

프로젝트 루트의 가상환경을 사용해 아래와 같은 방식으로 직접 실행했습니다.

```bash
cd ElXGB
../.venv/bin/python -u - <<'PY'
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from core.elxgb_classifier import ELXGBClassifier

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

split_idx = 15
X_train_list = [X_train[:, :split_idx], X_train[:, split_idx:]]
feat_names_list = [feature_names[:split_idx], feature_names[split_idx:]]

model = ELXGBClassifier(
    n_estimators=2,
    max_depth=2,
    eps=1.0 / 32,
    learning_rate=1.0,
    dp_epsilon=10.0,
    num_passive_parties=2,
)
model.fit(X_train_list, y_train, feat_names_list)
PY
```

## Results

| Metric | Value |
| --- | ---: |
| Train accuracy | 0.9472 |
| Test accuracy | 0.9298 |
| Pure train time (sec) | 81.3569 |
| Wall-clock train time (sec) | 85.8284 |
| Total communication (bytes) | 958,344 |
| Total communication (KB) | 935.8828 |
| Total communication (MB) | 0.9139 |

## Notes

- `pure_train_time_sec`는 코드 내부에서 측정한 순수 학습 연산 시간입니다.
- `train_wall_time_sec`는 모델 생성부터 `fit()` 종료까지의 실제 경과 시간입니다.
- `total_comm_bytes`는 `fit()` 내부에서 누적한 broadcast/histogram/local-gain 전송량입니다.
- 이번 실행은 빠르게 재현 가능한 로컬 내장 데이터셋(`breast_cancer`) 기준입니다.

## Raw Metrics File

같은 실행 결과는 JSON으로도 저장했습니다.

- [training_run_metrics.json](/Users/jachoi/Projects/codex/ElXGB/docs/training_run_metrics.json)
