import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from crypto.heservice import HEService
from core.active_party import ActiveParty
from core.passive_party import PassiveParty

def test_encrypted_histograms():
    print("Testing ELXGB Algorithm 1: HENS - Encrypted Histogram Computation...")
    
    # 1. 인프라 세팅 (HE 컨텍스트, Parties 초기화)
    he_svc = HEService()
    active_party = ActiveParty(he_svc)
    passive_party = PassiveParty(eps=0.5) # 피처 당 2개의 버킷 생성 지시
    
    # 2. 데이터 세팅 (N=4)
    # Active Party (y 레이블 할당 - 기댓값이 0이 되지 않도록 비대칭 배치)
    y_true = np.array([1, 1, 0, 0])
    active_party.set_data(y_true)
    active_party.initialize_predictions() # 기본 예측치(0.5) 초기화
    
    # Passive Party (X 피처 - 1차원 데이터, 0~50 범위)
    X_mock = np.array([
        [10.0],  # Bin 0 (0 <= x < 25)
        [20.0],  # Bin 0 
        [40.0],  # Bin 1 (25 <= x <= 50)
        [50.0]   # Bin 1
    ])
    passive_party.set_data(X_mock)
    
    # 3. 로컬 버킷(I 배열) 맵핑 추출
    passive_party.generate_global_buckets()
    print(f"Buckets (I binary masks): {[v.tolist() for v in passive_party.buckets[0]]}")
    
    # 4. HENS Step 1: Active Party가 암호화된 그래디언트를 Passive Party 측으로 보냄 (모사)
    g_raw, h_raw = active_party._compute_raw_gradients()
    enc_g, enc_h = active_party.compute_and_encrypt_gradients()
    print(f"Raw Gradients Computed by ActiveParty: {g_raw.tolist()}")
    
    # 5. HENS Step 2: Passive Party가 암호화된 히스토그램 연산 (HE Dot-product)
    print("\n--- [Passive Party: Computing Homomorphic Dot Products] ---")
    encrypted_histograms = passive_party.compute_encrypted_histograms(enc_g, enc_h)
    bin_histograms_feature_0 = encrypted_histograms[0]
    
    # 6. [Verification Agent 검증] 실제 수신자인 Active Party가 값을 받아 복원(Decrypt) 시 일치율 검사
    print("\n--- [Active Party: Decrypting Histograms for Verification] ---")
    for b_idx, (enc_sum_g, enc_sum_h) in enumerate(bin_histograms_feature_0):
        # dot product의 결과물은 스칼라이지만, CKKS 복호화 인터페이스 특성상 [scalar] 형태의 리스트 반환
        dec_sum_g = he_svc.decrypt(enc_sum_g)[0]
        
        # [검증 대비용] 평문 수동 스칼라 합 
        plain_mask = passive_party.buckets[0][b_idx]
        expected_sum_g = np.dot(g_raw, plain_mask)
        
        print(f"Bin {b_idx} Decrypted Sum(g) : {dec_sum_g:.4f} \t Expected: {expected_sum_g:.4f}")
        
        # HE 특성상 소수점 정밀도 1e-3 통과 여부 검수
        assert abs(dec_sum_g - expected_sum_g) < 1e-3, f"Bin {b_idx} Homomorphic Dot Product Failed! Privacy/Integrity Broken."
        
    print("\nTest Passed: Passive Party successfully generated encrypted histograms via HE dot-products without ever seeing raw gradients.")

if __name__ == "__main__":
    test_encrypted_histograms()
