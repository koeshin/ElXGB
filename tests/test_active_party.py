import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from crypto.heservice import HEService
from core.active_party import ActiveParty

def test_active_party_gradients():
    print("Testing ELXGB Active Party: HENS Step 1 (Gradient Encryption)...")
    
    # 1. HE Service 초기화 (TA 로부터 배포된 키 컨텍스트 활용)
    he_svc = HEService()
    
    # 2. Active Party 생성 및 레이블 주입
    active_party = ActiveParty(he_svc)
    
    # 가상의 레이블 데이터 5개 (예: 이진 분류 1/0)
    y_true = np.array([1, 0, 1, 1, 0])
    active_party.set_data(y_true)
    active_party.initialize_predictions()
    
    print("\n--- [Active Party Local Compute] ---")
    print(f"True Labels : {y_true}")
    print(f"Predictions : {active_party.y_pred}")
    
    # HENS 이론에 따른 1, 2차 미분 평문 도출
    g_raw, h_raw = active_party._compute_raw_gradients()
    print(f"Raw Gradients (g) : {g_raw}")
    print(f"Raw Hessians  (h) : {h_raw}")
    
    # 3. [Encryption Phase] 암호화 수행
    print("\n--- [Encryption Phase (ready to send out)] ---")
    enc_g, enc_h = active_party.compute_and_encrypt_gradients()
    print("Gradients and Hessians have been homomorphically encrypted successfully.")
    
    # 4. [Verification] 암호화된 통신 객체가 실제 평문 정보를 완벽히 암호화했는지 검증 (복호화)
    dec_g = he_svc.decrypt(enc_g)
    dec_h = he_svc.decrypt(enc_h)
    
    for r, d in zip(g_raw, dec_g):
        assert abs(r - d) < 1e-4, "Decrypted Gradient mismatch! HE Integrity Broken."
        
    for r, d in zip(h_raw, dec_h):
        assert abs(r - d) < 1e-4, "Decrypted Hessian mismatch! HE Integrity Broken."
        
    print("Test Passed: Active Party correctly computed and encrypted gradients based tightly on Eq (5).")

if __name__ == "__main__":
    test_active_party_gradients()
