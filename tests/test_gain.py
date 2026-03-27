import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from crypto.heservice import HEService
from core.active_party import ActiveParty
from core.passive_party import PassiveParty

def test_optimal_split_gain():
    print("Testing ELXGB Algorithm 1: HENS - Active Party Optimal Split Gain Search...")
    
    # 1. 인프라 준비
    he_svc = HEService()
    active_party = ActiveParty(he_svc)
    
    # Party A와 Party B를 생성하여 모델에 피처 분산 배정
    passive_party_A = PassiveParty(eps=0.5)  # 2 Bucket (Bin)
    passive_party_B = PassiveParty(eps=0.5)  # 2 Bucket (Bin)
    
    # 2. 데이터 세팅 (이진 분류라 가정)
    y_true = np.array([1, 1, 0, 0])
    active_party.set_data(y_true)
    active_party.initialize_predictions()
    
    # [Party A 피처 세팅]
    # 실제 값에 따라 완벽히 분류 가능한 피처 (y=1은 Bin 0으로, y=0은 Bin 1로 완전 분리 상태)
    X_mock_A = np.array([[10], [10], [50], [50]])  
    passive_party_A.set_data(X_mock_A)
    passive_party_A.generate_global_buckets()
    
    # [Party B 피처 세팅]
    # 불필요한 Noise 피처 (분리기준이 불분명하여 Information Gain이 낮을 것으로 예상)
    X_mock_B = np.array([[10], [50], [10], [50]])  
    passive_party_B.set_data(X_mock_B)
    passive_party_B.generate_global_buckets()
    
    # 3. Active -> Passive 로 기울기 전송 (암호화)
    enc_g, enc_h = active_party.compute_and_encrypt_gradients()
    
    # 4. Passive Parties가 은밀하게 동형암호 점곱 히스토그램 연산 후 회신
    hist_A = passive_party_A.compute_encrypted_histograms(enc_g, enc_h)
    hist_B = passive_party_B.compute_encrypted_histograms(enc_g, enc_h)
    
    encrypted_histograms_per_party = {
        'Party_A': hist_A,
        'Party_B': hist_B
    }
    
    # 5. Active 측에서 암호문을 회수, 복호화하여 최고 품질의 트리를 만들기 위해 식(Eq 1~4) 가동
    print("\n--- [Active Party: Finding Optimal Gain] ---")
    best_split, max_gain = active_party.calculate_optimal_split(encrypted_histograms_per_party, lambda_val=1.0)
    
    print(f"Optimal Split Decision -> Party: {best_split[0]}, Feature: {best_split[1]}, Bin Index: {best_split[2]}")
    print(f"Max Information Gain -> {max_gain:.4f}")
    
    # 6. [Verification Agent 검증] 
    # 수학적으로 노이즈(Party_B)가 아닌 패턴(Party_A)이 완벽한 게인(Gain)을 가짐을 통과해야 함.
    assert best_split[0] == 'Party_A', "Gain Calculation failed! Did not logically pick the mathematically optimal splitting party."
    
    print("\nTest Passed: Active Party successfully decrypted histograms and found the optimal split mathematically following Eq (1-4).")

if __name__ == "__main__":
    test_optimal_split_gain()
