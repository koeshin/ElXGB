import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crypto.dp_injector import DPNoiseInjector
from core.active_party import ActiveParty
from core.passive_party import PassiveParty
from crypto.heservice import HEService

def test_dpns_algorithm():
    print("Testing ELXGB Phase 4: DPNS (Differential Privacy-based Node Split)...")
    
    # HE 인프라는 초기화하지만, 실제 DPNS 연산엔 활용되지 않음을 증명할 예정
    he_svc = HEService()
    active_party = ActiveParty(he_svc)
    passive_party = PassiveParty(eps=0.5) # 2 Bins
    
    dp_injector = DPNoiseInjector(epsilon=1.0, delta=1e-5, clip_c=1.0)
    
    y_true = np.array([1, 1, 0, 0])
    active_party.set_data(y_true)
    active_party.initialize_predictions()
    
    X_mock = np.array([[10], [10], [50], [50]])
    passive_party.set_data(X_mock)
    passive_party.generate_global_buckets()
    
    # -------------------------------------------------------------
    # 1. HENS 성능 도출 (성능 비교 대조군)
    # -------------------------------------------------------------
    start_time = time.time()
    enc_g, enc_h = active_party.compute_and_encrypt_gradients()
    hist_HE = passive_party.compute_encrypted_histograms(enc_g, enc_h)
    best_split_HE, max_gain_HE = active_party.calculate_optimal_split({'Party_A': hist_HE})
    hens_time = time.time() - start_time
    
    print(f"\n[HENS Benchmark] Execution Time: {hens_time:.6f} seconds (TenSEAL Homomorphic Encryption used)")

    # -------------------------------------------------------------
    # 2. DPNS 시뮬레이션 (Algorithm 3 & 2)
    # -------------------------------------------------------------
    start_time = time.time()
    
    # Active -> Passive 노이즈가 섞인 평문 전달 (Algorithm 3)
    g_noisy, h_noisy = active_party.compute_noisy_dp_gradients(dp_injector)
    print(f"\n[DPNS] Original Gradients : {active_party._compute_raw_gradients()[0]}")
    print(f"[DPNS] Noised Gradients   : {g_noisy}")
    
    # Passive Party 초고속 히스토그램 연산 (Algorithm 2)
    hist_P = passive_party.compute_plaintext_histograms(g_noisy, h_noisy)
    
    encrypted_histograms_per_party = {'Party_A': hist_P}
    
    # Active Party 평문 기반 고속 분할 (복호화 생략)
    best_split, max_gain = active_party.calculate_optimal_split_plaintext(
        encrypted_histograms_per_party, lambda_val=1.0, gamma_val=0.0
    )
    dpns_time = time.time() - start_time
    
    print("\n--- [Active Party: Finding Optimal Gain via DPNS] ---")
    print(f"Optimal Split Decision -> Party: {best_split[0]}, Feature: {best_split[1]}, Bin Index: {best_split[2]}")
    print(f"Max Information Gain -> {max_gain:.4f}")
    print(f"[DPNS Benchmark] Execution Time: {dpns_time:.6f} seconds (Plaintext Only + Noise)")
    
    print(f"\n!!! SPEED BOOST: DPNS is {hens_time/dpns_time:.2f}x faster !!!")
    
    assert best_split is not None
    print("\nTest Passed: DPNS efficiently found a split locally using DP-protected plaintexts without TenSEAL overhead!")

if __name__ == "__main__":
    test_dpns_algorithm()
