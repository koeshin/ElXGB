import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from core.passive_party import PassiveParty

def test_passive_party_buckets():
    print("Testing ELXGB Passive Party: Bucket Generation (Global Proposal)...")
    
    # 1. 1/eps = 4 개의 버킷 생성을 지시 (eps=0.25)
    eps = 0.25
    passive_party = PassiveParty(eps=eps)
    
    # 2. 가상의 피처 데이터 세팅: 5개의 샘플(N=5), 2개의 피처(m=2)
    # Feature 0: 10~50, Feature 1: 0.1~0.9
    X_mock = np.array([
        [10.0, 0.1],
        [20.0, 0.9],
        [30.0, 0.2],
        [40.0, 0.8],
        [50.0, 0.5]
    ])
    passive_party.set_data(X_mock)
    
    print(f"\n--- [Passive Party Local Compute] ---")
    print(f"Features (X) Shape: {X_mock.shape}")
    print(f"Target Buckets per Feature: {int(1.0/eps)}")
    
    # 3. 버킷 및 이진 벡터(I) 생성 알고리즘 실행
    print("\n--- [Generating Global Buckets (Binary Vectors I)] ---")
    passive_party.generate_global_buckets()
    
    # 결과 이진 배열 시각화
    for attr_idx, vectors in passive_party.buckets.items():
        print(f"\nFeature {attr_idx} Buckets:")
        for b_idx, I_vec in enumerate(vectors):
            print(f" Bin {b_idx} (I vector): {I_vec.tolist()}")
    
    # 4. [Verification] 각 데이터 항목이 상호 배타적으로 "오직 1개의 버킷(Bin)"에만 속해야 함
    n_samples, m_features = X_mock.shape
    
    for attr_idx, vectors in passive_party.buckets.items():
        # 모든 버킷을 합쳤을 때 N차원 벡터는 전부 [1, 1, 1, 1, 1] 이어야만 정보 손실과 중복이 없음.
        sum_across_bins = np.sum(vectors, axis=0)
        
        for s_idx, s_val in enumerate(sum_across_bins):
            assert s_val == 1, f"Sample {s_idx} in Feature {attr_idx} is in {s_val} bins!"
            
    print("\nTest Passed: All features correctly converted into disjoint Binary Vectors (I).")

if __name__ == "__main__":
    test_passive_party_buckets()
