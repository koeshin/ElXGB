# Main vs Jina Changes

## Overview

이 문서는 `main` 브랜치 대비 `jina` 브랜치에서 어떤 코드가 어떻게 바뀌었는지 정리한 요약 문서입니다.

비교 기준:

- Base branch: `main`
- Target branch: `jina`

## High-Level Summary

`jina` 브랜치의 변경은 크게 네 가지입니다.

1. DPNS 학습 프로토콜을 passive party의 local best split 방식으로 수정
2. 학습 과정에 attribute obfuscation 메타데이터를 추가
3. 학습 결과 구조에서 평문 split 메타데이터를 제거
4. 학습 경로와 시간/통신량 측정을 검증하는 unit test 및 실행 보고서 추가

## File-Level Changes

### 1. [core/elxgb_classifier.py](/Users/jachoi/Projects/codex/ElXGB/core/elxgb_classifier.py)

주요 변경 사항:

- `self.num_passive_parties` 필드를 명시적으로 저장
- 전체 feature 수(`self.total_feature_count`)를 계산해서 passive party에 전달
- 각 passive party에 대해 global feature slot 정보를 부여
- DPNS 단계에서 전체 histogram을 active로 보내는 방식 대신, passive party가 계산한 local best candidate만 active가 수집하도록 변경
- 트리 노드에 더 이상 평문 `split`, `split_condition`를 저장하지 않도록 변경
- 평문 추론 경로에서 더 이상 `is_flipped`를 사용하지 않도록 정리

의미:

- `main`보다 `jina`가 논문의 DPNS 통신 구조에 더 가깝습니다.
- 학습 결과 트리에 평문 split 정보가 직접 남지 않도록 바꿨습니다.

### 2. [core/passive_party.py](/Users/jachoi/Projects/codex/ElXGB/core/passive_party.py)

주요 변경 사항:

- `global_feature_slots`, `total_feature_count` 필드 추가
- `set_data(...)`가 global slot 메타데이터를 받을 수 있도록 확장
- `find_best_split_plaintext(...)` 추가
- `register_obfuscated_split(...)`가 `threshold_vector`를 생성하도록 수정
- 기존 `is_flipped` 기반 direction obfuscation 데이터 저장 제거

의미:

- `main`에서는 passive party가 DPNS에서 전체 histogram을 active에 전달했지만, `jina`에서는 split 후보 1개만 반환합니다.
- attribute obfuscation을 위해 feature slot과 threshold를 감춘 벡터를 생성합니다.

### 3. [crypto/heservice.py](/Users/jachoi/Projects/codex/ElXGB/crypto/heservice.py)

주요 변경 사항:

- `PaillierVector.__add__` 추가
- `encrypt_scalar(...)` 추가
- `decrypt(...)`가 `PaillierVector` 자체도 복호화할 수 있도록 확장
- `decrypt_scalar(...)` 추가

의미:

- attribute obfuscation 과정에서 스칼라 단위 Paillier 암복호화가 필요해서 지원 기능을 추가했습니다.

## Algorithmic Difference Summary

### DPNS

`main`:

- passive party가 noisy plaintext histogram 전체를 active로 전송
- active가 모든 feature/bin을 중앙에서 직접 스캔

`jina`:

- passive party가 local best split candidate 1개만 계산
- active는 각 party가 보낸 후보들 중 최대 gain만 선택

즉, `jina`는 논문의 “local max gain only” 흐름에 더 가깝습니다.

### Attribute Obfuscation

`main`:

- 트리 노드에 `split`, `split_condition`가 평문으로 남음
- passive party의 로컬 레코드에는 threshold와 `is_flipped`가 저장됨

`jina`:

- 트리 노드에서 `split`, `split_condition` 제거
- passive party가 `threshold_vector`를 생성해 obfuscation metadata를 유지

즉, `jina`는 학습 산출물에서 평문 split 정보 노출을 줄였습니다.

### Inference

`main`:

- 평문 threshold 기반 추론
- offline inference helper 포함

`jina`:

- 요청사항에 맞춰 추론은 평문 경로 유지
- 학습 쪽 수정과 맞지 않는 `is_flipped` 의존성만 제거

즉, `jina`는 추론을 새 보안 프로토콜로 확장하지 않고, 학습 프로토콜 수정에 집중했습니다.

## Tests Added on `jina`

추가된 테스트 파일:

- [test_training_pipeline.py](/Users/jachoi/Projects/codex/ElXGB/tests/test_training_pipeline.py)
- [test_attribute_obfuscation_unit.py](/Users/jachoi/Projects/codex/ElXGB/tests/test_attribute_obfuscation_unit.py)
- [test_dpns_protocol_unit.py](/Users/jachoi/Projects/codex/ElXGB/tests/test_dpns_protocol_unit.py)

각 테스트가 확인하는 내용:

- 학습이 실제로 끝까지 동작하는지
- 평문 `predict()`가 정상 동작하는지
- `total_pure_train_time`, `total_comm_bytes`가 채워지는지
- 내부 노드에 평문 split 정보가 남지 않는지
- attribute obfuscation용 `threshold_vector`가 생성되는지
- DPNS에서 candidate 1개만 반환하는지

## Measured Training Run

`jina`에서는 실제 데이터셋 실행 결과도 함께 정리했습니다.

- [training_run_report.md](/Users/jachoi/Projects/codex/ElXGB/docs/training_run_report.md)
- [training_run_metrics.json](/Users/jachoi/Projects/codex/ElXGB/docs/training_run_metrics.json)

이 실행은 `sklearn breast_cancer` 데이터셋 기준이며, 학습 시간과 communication cost를 포함합니다.

## Practical Conclusion

정리하면 `jina`는 `main`의 실행 가능한 코드를 바탕으로 다음 방향으로 수정한 브랜치입니다.

- DPNS를 중앙 histogram 스캔 방식에서 local best split 방식으로 변경
- attribute obfuscation 메타데이터 추가
- 학습 결과 구조에서 평문 split 정보 제거
- 위 변경을 검증하는 테스트와 실행 보고서 추가

즉, `jina`는 `main`보다 논문의 학습 알고리즘 쪽에 더 가깝게 정리된 브랜치입니다.
