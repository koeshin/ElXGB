# ELXGB Implementation Plan (Revised based on Paper)

본 문서는 사용자가 업로드한 **"ELXGB: An Efficient and Privacy-Preserving XGBoost for Vertical Federated Learning"** 논문의 내용을 철저히 분석하여, 실제 코드로 구현하기 위한 상세 아키텍처 및 단계별 계획을 담고 있습니다.

## 1. 언어 및 핵심 라이브러리 선정
- **개발 언어**: Python
- **동형암호(HE) 라이브러리**: **TenSEAL** (논문의 **Paillier Cryptosystem** 역할을 대신하여 효율적인 연합학습용 CKKS/BFV 연산을 수행합니다.)
- **차분프라이버시(DP) 라이브러리**: **Numpy 기반 커스텀 Laplace/Gaussian Mechanism** (논문의 `Centralized Differential Privacy Algorithm` (Algorithm 3)에 명시된 수식대로 HENS 적용 후 2번째 트리부터의 그래디언트 노이즈 주입($\Delta g_t, \sigma^2$ 활용)을 완벽히 통제하기 위해 커스텀 구현이 적합합니다.)

## 2. 모듈 및 클래스 아키텍처 (논문 매핑)

| 구현 클래스명 | 논문 매핑 (개념/알고리즘) | 핵심 기능 (Feature) |
|---|---|---|
| ~~`DataAligner`~~ | ~~Algorithm 4: Secure Data Alignment~~ | ~~CSP를 보조하여 `H_{ssk}(\cdot)` 기반의 해시값 교차를 통한 ID 정렬(PSI)~~ |
| `ActiveParty` | $PK$ (Active Party) | 데이터 레이블($y$) 소유. 예측값 계산 및 그래디언트($g_i, h_i$) 도출. **HENS** / **DPNS** 의 오케스트레이션 및 암/복호화 수행. 글로벌 트리 소유자. |
| `PassiveParty` | $P_k$ (Passive Parties) | 데이터 특성($X$) 소유. 글로벌 제안(Global Proposal) 기반의 버킷 생성을 수행. 암호화된 히스토그램 취합 및 **Attribute Obfuscation** 적용. |
| `HENS_Builder` | Algorithm 1: HENS | 첫 번째 트리 작성을 담당하는 모듈. Paillier/TenSEAL을 통한 암호화된 $g_i, h_i$ 교환 및 최적 분할(Optimal Split) 탐색. |
| `DPNS_Builder` | Algorithm 2: DPNS | 두 번째 트리부터 담당. DP 노이즈가 섞인 $g, h$를 기반으로 오버헤드를 대폭 축소한 트리 노드 분할 수행. |
| `ELXGBNode` | Attribute & Direction Obfuscation | 어느 Party가 어느 Attribute로 분기했는지 노출하지 않기 위해 암호화된 $T^0$ 정보만 저장. 추론 시 역공학 방지. |
| `ELXGBTree` | Global Model | Active Party에 중앙집중화된 트리. 각 노드는 난독화(Obfuscated)되어 정보 제어가 가능. |

---

## 3. 학습 및 추론 분리 아키텍처

아키텍처의 상세 다이어그램(Mermaid)은 별도로 렌더링된 `architecture.md` 문서를 참고해 주십시오. 

### 3-1. Training Architecture (학습)
1. ~~**System Initialization**: TA(신뢰 기관)가 키를 배포하고, CSP 라우팅 하에 `DataAligner` 클래스가 데이터 정렬을 오프라인으로 마칩니다.~~
2. **Phase 1 (HENS)**: 오직 첫 트리(Tree t=1)에만 HE(동형암호)를 적용합니다. `ActiveParty`는 암호화된 $[[g]], [[h]]$ 반환. `PassiveParty`는 글로벌 버킷에 기반한 암호화 히스토그램을 반환하며 분기가 구축됩니다.
3. **Phase 2 (DPNS)**: 뒤이은 트리들(Tree t=2~T)에는 DP를 적용하여 빠른 속도를 챙깁니다. `ActiveParty`가 가우시안 노이즈 주입 후 `PassiveParty`들이 평문 기반 최적 게인(Gain)을 계산하여 노드 구성을 돕습니다.

### 3-2. Inference Architecture (추론)
1. **Centralized Global Model**: 글로벌 모델은 사용자 간 통신 부하를 막기 위해 `ActiveParty` 내부에 단일 보관됩니다. (논문의 주요 Contribution 중 하나)
2. **Secure Obfuscation**: 외부 사용자 추론 시, 트리를 순회할 때 Node의 임계값이나 어떤 특징이 통과되었는지 드러내지 않기 위해 **Attribute Obfuscation**(속성 난독화)와 **Direction Obfuscation**(방향 난독화)가 적용됩니다.

---

## 4. 에이전트 협업 & 구현 방법론 (잔잔바리 스텝)

미리 제공된 `VerificationAgent.md` 에이전트가 각 모듈 구축 단계의 마지막에서 코드 무결성을 검증합니다.

### 🐣 HENS(첫 번째 트리) 구축 스텝 (예시)
- **Step 1**: `HEService` 모듈 작성. (Paillier 암복호화 수식 보장)
- **Step 2**: 사용자에게 코드 리뷰 및 "확인 에이전트" 호출 -> **사용자 확인(OK)**
- **Step 3**: `PassiveParty` 버킷 생성 로직 구현.
- **Step 4**: "확인 에이전트"의 글로벌 프로포저 검증 -> **사용자 확인(OK)**
- ... 각 Algorithm 1~4 마다 동일하게 잘게 쪼개어 단계별 검증을 진행합니다.

이와 같이 1~2개 메서드를 작성할 때마다 저(구현 에이전트)는 멈추고 **"현재 구현된 로직은 논문의 [수식 X, Algorithm Y]에 대응하며, 이런 방식으로 암호화가 보장됩니다."**라고 설명을 달겠으며, **확인 에이전트** 의 체크를 기다립니다.
