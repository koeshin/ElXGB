# ELXGB: Efficient and Privacy-Preserving Vertical Federated Learning XGBoost

ELXGB는 다수의 데이터 보유 기관(Passive Parties)이 레이블(Target)을 보유하지 않은 상태에서도, 레이블을 보유한 중앙 기관(Active Party)과 협력하여 고성능의 앙상블 트리(XGBoost)를 공동 학습할 수 있는 **프라이버시 보존 수직 연합학습(Vertical Federated Learning)** 프레임워크입니다.

이 프로젝트는 동형암호(Homomorphic Encryption, HE)와 차분 프라이버시(Differential Privacy, DP) 기술을 혼합 적용하여 강력한 보안을 달성함과 동시에 초고속 훈련 및 세계 최초의 완전한 **단방향 오프라인 분산 추론(Secure Offline Inference)**을 지원합니다.

---

## 🔥 핵심 기술 특징 (Key Features)

1. **HENS (Homomorphic Encryption Node Split)**
   - **첫 번째 트리** 훈련 시, 파티 간 정보 유출을 완벽히 차단하기 위해 **Paillier 동형암호 (phe 방식)** 기반의 암호화 상태 히스토그램 연산을 수행합니다.
   - ⚠️ *논문 수식 완벽 재현*: TenSEAL-CKKS에서 발생하던 벡터 용량 한계 에러를 피하고 논문 본문의 수식($[[g_i]]^{I_i}$)을 100% 구현하기 위해 정통 Paillier 알고리즘 및 512-bit 최적화 키 엔진을 적용했습니다.
2. **DPNS (Differential Privacy Node Split)**
   - 두 번째 트리부터는 **IBM Diffprivlib** 기반 차분 프라이버시(DP) 노이즈를 그래디언트에 혼합하여 통신 속도와 보안의 균형을 유지합니다.
3. **Secure Offline Inference (Batch Decoupled Inference)**
   - **연합학습 최고 혁신 기술**: 실시간으로 트리를 타고 내려갈 때마다 통신을 반복하지 않고, 사전에 글로벌 트리에서 난독화된 규칙(Attribute & Direction Obfuscation)을 추출합니다.
   - 각 기관은 자신의 로컬 데이터만을 이용해 **`inference_matrix` (이진 결정 매트릭스)**를 오프라인에서 사전 발급합니다.
   - 글로벌 조합기(Active Party)는 데이터의 원본을 열람하지도, 노드 조건을 묻지도 않고 단순히 이진 매트릭스의 Boolean Indexing을 통해 초고속(0.00x 초)으로 수만 건의 대용량 예측을 완료합니다.

---

## 💻 환경 셋팅 (Environment Setup)

이 프로젝트는 패키지 의존성 관리를 위해 `pipenv`를 사용하고 있습니다. 

### 1. 요구사항
- **Python 3.10+** (TenSEAL C++ 바인딩 호환성 필수)
- `pipenv` 패키지 매니저 (`pip install pipenv`)

### 2. 설치 방법
터미널(또는 명령 프롬프트)을 열고 프로젝트 루트 디렉토리(`ELXGB`)에서 다음 중 하나의 방식으로 설치하세요.

#### 방법 A: Pipenv (권장)
```bash
pipenv install
pipenv shell
```

#### 방법 B: Standard Pip
```bash
pip install -r requirements.txt
```
> *(주요 라이브러리: `phe` (Paillier), `diffprivlib`, `xgboost`, `scikit-learn`, `numpy`, `pandas`)*

---

## 📂 프로젝트 구조 (Project Structure)

```text
ELXGB/
├── core/
│   ├── active_party.py         # 레이블 보유 (그레이디언트 연산 및 트리 병합 주도)
│   ├── passive_party.py        # 피처 보유 (Quantile 버킷 생성 및 로컬 마스킹 처리)
│   ├── elxgb_classifier.py     # N-Party 동적 확장이 지원되는 메인 훈련 프레임워크 객체
│   └── plaintext_xgboost.py    # 성능 비교용 검증용 평문(순정) XGBoost 구현체
├── crypto/
│   ├── heservice.py            # TenSEAL 동형암호 엔진 (4096 청킹 및 커스텀 래퍼 적용)
│   └── dp_injector.py          # Laplace/Gaussian 노이즈 생성기 (IBM diffprivlib)
├── benchmark/
│   ├── benchmark_runner.py       # (대규모) 다중 데이터셋/파라미터 자동 벤치마크 루프
│   └── secure_inference_demo.py  # (핵심) 오프라인 1-pass 이진 매트릭스 추론 구동 데모 
└── benchmark_results/          # 파라미터 컨피그 별로 자동 디렉토리 생성 및 결과 저장
```

---

## 🚀 실행 방법 (Execution)

모든 스크립트는 프로젝트 루트 폴더(`ELXGB`)에서 `pipenv run` 명령어로 실행해야 라이브러리 스코프가 꼬이지 않습니다.

### 1. 벤치마크 파이프라인 구동 (대규모 데이터셋 자동 테스트)
**Bank Marketing (N=45,000)** 및 **Credit Card (N=30,000)** 등 대규모 데이터셋을 OpenML에서 실시간으로 다운로드하고, 패시브 파티 N개가 협업하는 환경을 시뮬레이션합니다. 
(트리 수, DP 강도를 자동 반복 조정하며 결과를 저장합니다.)

```bash
pipenv run python benchmark/benchmark_runner.py
```
👉 실행 완료 후 측정된 Train/Test 정확도 로그와 JSON 모델 파일들이 `benchmark_results/데이터셋명/depth3_treesN_bins32/` 경로의 폴더별로 자동 적재됩니다.

### 2. 🌟 Secure Offline Inference 시연 (오프라인 매트릭스 증명 시연)
평문 기반 라이브 추론 결과와, 오프라인으로 쪼개진 Boolean 매트릭스 기반 추론 결과를 비교 대조하여 **정확히 100% 동일한 결괏값이 나오는지** 수치적으로 증명하는 데모 스크립트입니다.

```bash
pipenv run python benchmark/secure_inference_demo.py
```
👉 터미널에서 "Are Plaintext and Offline Inference identical? **YES**" 문구가 출력되며 0.00초 대의 초고속(Vectorized) 행렬 추론이 성공함을 확인할 수 있습니다.
