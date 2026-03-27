# ELXGB Implementation Task List (Based on Paper)

## 1. 아키텍처 및 환경 설정 (PLANNING)
- [x] 논문 원본(PDF) 텍스트 추출 및 알고리즘 분석 완료
- [x] [Architecture Agent] 논문 기반 훈련/추론 아키텍처(Mermaid) 그리기 및 docs/ 폴더 세팅
- [x] 상세 클래스 및 역할 매핑 확정 (HENS, DPNS 알고리즘 반영)

## 4. DPNS (두 번째 트리 이후의 비용 효율적 분할) 파이프라인 (EXECUTION)
- [x] ~~`ActiveParty`의 DP 적용 노이즈 그레디언트 공유 (Algorithm 3)~~
- [x] ~~`PassiveParty` 측 평문 히스토그램 연산 구현 (Algorithm 2)~~
- [x] ~~`ActiveParty` 평문 기반 고속 최적 분할 모듈 추가 구현~~
- [x] ~~[Verification Agent] DPNS 단위 속도 벤치마크 테스트 및 검수 승인~~

## 5. 트리의 보안 난독화 파이프라인 (EXECUTION)
- [x] ~~DP 발생기를 신뢰성 있는 IBM `diffprivlib`로 전면 교체 완료~~
- [x] ~~`PassiveParty`의 로컬 난독화 해시(`record_id`) 및 방향(Flip) 발급 매핑 구현 완료~~
- [x] ~~[요청 사항] 트리 가시화를 위해 `split`, `split_condition` (평문 피처/임계치) 복원 추가~~

## 6. HENS + DPNS 하이브리드 부스팅 벤치마킹 (VERIFICATION)
- [/] ~~[진행중] Tree 1번(HENS) + Tree N번(DPNS) 혼합 부스팅 오케스트레이션 적용~~
- [/] ~~[진행중] 3-Tree (Depth 3) XGBoost vs ELXGB 정면 정확도 벤치마크 구동~~
- [ ] 파이프라인 안전성/정확도 전면 승인 및 문서화 래퍼 `HEService` (TenSEAL) 구현 및 단위 테스트~~
- [x] ~~DP 노이즈 주입 클래스 구현 (Algorithm 3) 및 감도(Sensitivity) 대응 로직 테스트~~
- [x] ~~[Verification Agent] 암호화 모듈의 무결성 승인~~

## 3. HENS (첫 번째 트리의 비용 높은 분할) 기반 코어 모델 (EXECUTION)
- [x] ~~`ActiveParty`의 암호화된 그래디언트 $[[g_i]], [[h_i]]$ 연산 함수 구현~~
- [x] ~~`PassiveParty`의 로컬/글로벌 버킷(Bucket) 생성 로직 구현~~
- [x] ~~`PassiveParty`의 암호화된 히스토그램 도출 (Algorithm 1) 구현~~
- [x] ~~`ActiveParty` 측 수신 히스토그램 복호화 및 최적 분할(Gain) 탐색 로직 (Eq 1~4) 구현~~
- [x] ~~[Verification Agent] HENS 수식 (Eq 1~4) 및 노드 생성 일치도 검수 후 승인~~
- [x] ~~트리 루트~리프 전역 오케스트레이션(첫 번째 트리 완전 구동) 작성~~
- [x] ~~[Verification Agent] Breast Cancer 데이터 + 2 Passive Party 셋업으로 XGBoost 정식 라이브러리 트리와 1:1 성능/구조 비교~~

## 4. DPNS (두 번째 트리 이후의 비용 효율적 분할) 파이프라인 (EXECUTION)
- [x] ~~`ActiveParty`의 DP 적용 그레디언트 공유(Algorithm 3)~~
- [x] ~~`PassiveParty`의 평문 게인 계산 및 반환 (Algorithm 2)~~
- [x] ~~[Verification Agent] DPNS 프로세스 벤치마크 (TenSEAL 회피 가속) 승인~~

## 5. 트리의 보안 난독화 파이프라인 (EXECUTION)
- [x] ~~DP 발생기를 신뢰성 있는 IBM `diffprivlib`로 전면 교체 완료~~
- [x] ~~`PassiveParty`의 난독화 `record_id` 분발 및 방향(Flip) 발급 매핑 완료~~
- [x] ~~[사용자 요청] 트리 가시화를 위해 `split`, `split_condition` (평문) 복원 병행~~

## 6. HENS+DPNS 연쇄 3-Tree 부스팅 벤치마크 (VERIFICATION)
- [x] ~~Tree 1 (HENS) -> Tree 2, 3 (DPNS) 다중 트리 부스팅 오케스트레이션 연동~~
- [x] ~~[Verification Agent] XGBoost 대비 3-Tree (Depth 3) ELXGB 통합 벤치마크 승인 (Acc: 100%)~~

## 7. 최후의 보루: 안전한 추론 (Inference) 아키텍처 (FINAL)
- [x] ~~[사용자 지침] 오프라인 복호 체계를 제외하고 완전성 평문 추론 평가로 아키텍처 우회 결정~~
- [x] ~~Breast-Cancer 전체 데이터셋(N=569) 통째로 투입 (Train:Test = 80:20 분할)~~
- [x] ~~[Verification Agent] XGBoost 대비 3-Tree (Depth 4) 모델 예측(Test Acc) 완승 벤치마크 입증 및 최종 승인~~
