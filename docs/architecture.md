# ELXGB Architecture Diagram

논문 "ELXGB: An Efficient and Privacy-Preserving XGBoost for Vertical Federated Learning" 에 기반한 학습 및 추론 구조도입니다.

## 1. System Entities (시스템 개체)
- **TA (Trusted Authority)**: 키 생성 및 분배 관리자 (System Initialization)
- **CSP (Cloud Service Provider)**: 암호화된 상태의 Data Alignment(PSI) 및 추론(Inference) 라우팅 보조.
- **Active Party (PK)**: 데이터 레이블(Label $y$)을 소유하고 글로벌 중앙 집중형 트리를 구축 및 호스팅하는 주관사.
- **Passive Parties ($P_k$)**: 고유의 피처(Feature $X$)를 가지며, 히스토그램 연산을 수행하고 속성을 난독화하는 참여기관들.

## 2. Model Training Architecture (모델 학습)

```mermaid
sequenceDiagram
    participant TA as Trusted Authority
    participant CSP as Cloud Service Provider
    participant PP as Passive Parties (P1~PK-1)
    participant AP as Active Party (PK)
    
    %% Initialization
    rect rgb(240, 248, 255)
    Note over TA, AP: Phase 0. System Initialization & Data Alignment
    TA->>AP: 1. Generate & Distribute Keys (Paillier PK, ssk for HMAC)
    TA->>PP: 1. Distribute keys
    PP->>CSP: 2. Send Hashed IDs
    AP->>CSP: 2. Send Hashed IDs
    CSP-->>PP: 3. Intersect IDs (Aligned Data)
    CSP-->>AP: 3. Intersect IDs (Aligned Data)
    end
    
    %% HENS
    rect rgb(255, 245, 238)
    Note over PP, AP: Phase 1. HENS: HE-based Node Split (For the 1st Tree)
    AP->>AP: Calculate [[g]], [[h]] using Paillier Enc
    AP->>PP: Send encrypted [[g]], [[h]]
    PP->>PP: Update Buckets (Global Proposal) & Compute Encrypted Histograms
    PP->>AP: Send ⟨[[g_L]]⟩, ⟨[[h_L]]⟩
    AP->>AP: Decrypt & Compute Max Gain (find best split)
    AP->>PP: Request optimal Attribute Split construction
    PP->>PP: Apply Attribute Obfuscation (T^0)
    PP-->>AP: Send Obfuscated Node Info (Global Model Update)
    end

    %% DPNS
    rect rgb(240, 255, 240)
    Note over PP, AP: Phase 2. DPNS: DP-based Node Split (For Tree 2 ~ T)
    AP->>AP: Add Differential Privacy (DP) Noise to g, h
    AP->>PP: Send Noisy Gradients (~g, ~h)
    PP->>PP: Update Buckets & Compute local max gain
    PP->>AP: Send local max gains
    AP->>AP: Compute global max gain
    AP->>PP: Request corresponding split node
    PP->>PP: Apply Attribute Obfuscation
    PP-->>AP: Send Obfuscated Node Info (Global Model Update)
    end
```

## 3. Secure Inference Architecture (안전한 추론)

```mermaid
sequenceDiagram
    participant User
    participant CSP
    participant AP as Active Party (PK) Hosting Global Model
    participant PP as Passive Parties (For specific Attribute Info)

    Note over User, PP: Inference Phase (Centralized Global Model on PK)
    
    User->>CSP: Request Inference Service
    CSP->>AP: Forward request & obfuscated user data
    
    AP->>AP: Compare request data against Centralized Tree
    Note over AP: Apply 'Direction Obfuscation' to hide which branch<br/>is taken, protecting the model from inversion attacks.
    
    alt Needs passive party attribute info
        AP->>PP: Fetch matching Condition (Attribute Obfuscation)
        PP-->>AP: Condition result (True/False implicitly)
    end
    
    AP->>AP: Aggregate Leaf Weights
    AP-->>CSP: Sent Encrypted/Obfuscated Result
    CSP-->>User: Final Prediction Output
```
