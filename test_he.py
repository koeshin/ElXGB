from heservice import HEService

def test_he_service():
    print("Testing ELXGB Homomorphic Encryption (HENS Building Block)...")
    
    # 1. Initializing HE Service (Active Party 역할)
    he_svc = HEService()
    
    # 2. 평문 기울기/헤시안 Mockup 데이터
    g1 = [0.5, -0.3, 1.2, 0.0]
    g2 = [0.1,  0.4, -0.2, 0.5]
    
    print("Plaintext g1:", g1)
    print("Plaintext g2:", g2)
    
    # 3. [Active Party] 암호화 수행
    enc_g1 = he_svc.encrypt(g1)
    enc_g2 = he_svc.encrypt(g2)
    print("Encryption successful.")
    
    # 4. [Passive Party] 암호 상태에서 덧셈 (HENS의 히스토그램 집계 모사)
    enc_sum = he_svc.add(enc_g1, enc_g2)
    print("Homomorphic Addition successful.")
    
    # 5. [Active Party] 복호화 및 검증
    dec_sum = he_svc.decrypt(enc_sum)
    expected_sum = [g1[i] + g2[i] for i in range(len(g1))]
    
    print("Decrypted Sum:", [round(x, 4) for x in dec_sum])
    print("Expected Sum :", expected_sum)
    
    # CKKS 모델은 부동소수점 근사 암호이므로 약간의 오차 허용 (1e-3)
    for res, exp in zip(dec_sum, expected_sum):
        assert abs(res - exp) < 1e-3, "HE Addition Failed: Precision loss too high!"
        
    print("Test Passed: Additive Homomorphism securely executed using TenSEAL.")

if __name__ == "__main__":
    test_he_service()
