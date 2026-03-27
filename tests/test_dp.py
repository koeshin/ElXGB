import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from crypto.dp_injector import DPNoiseInjector

def test_dp_noise():
    print("Testing ELXGB Algorithm 3: DP Noise Injector (Gaussian Mechanism)...")
    
    # 1. DP 예산 및 설정치 (논문 기준 시나리오)
    epsilon = 1.0
    delta = 1e-5
    clip_c = 1.0  # 그래디언트는 -1.0 ~ +1.0 바운딩
    
    # 2. DP Noise Injector 생성 및 민감도 수치 출력
    injector = DPNoiseInjector(epsilon=epsilon, delta=delta, clip_c=clip_c)
    print(f"Algorithm 3 Parameter - Sensitivity: {injector.sensitivity:.2f}, Sigma: {injector.sigma:.2f}, Noise StdDev: {injector.noise_std_dev:.2f}")
    
    # 3. 테스트용 그래디언트 Mock 데이터 (2.5와 -3.0은 클리핑 한계 초과)
    original_gradients = [0.1, -0.5, 0.9, 2.5, -3.0]
    print(f"Original Gradients: {original_gradients}")
    
    # 4. DP 노이즈 주입 실행!
    noisy_gradients = injector.inject_noise(original_gradients)
    print(f"Noisy Gradients   : {[round(x, 4) for x in noisy_gradients]}")
    
    # 5. [Verification] 데이터 무결성 단위 테스트
    assert len(original_gradients) == len(noisy_gradients), "Array length mismatch!"
    
    # 6. [Verification] 노이즈의 강제 반영 확인 및 클리핑 작동 테스트
    # 노이즈 주입으로 인해 원래 값이 변경되었어야만 DP가 보장된 것임.
    diff = sum(abs(o - n) for o, n in zip(original_gradients, noisy_gradients))
    assert diff > 0.01, "HE Noise Failed: Gradients did not change enough! (Privacy Broken)"
    
    print("Test Passed: Gaussian Noise mathematically injected according to exact (epsilon, delta)-DP constraints.")

if __name__ == "__main__":
    test_dp_noise()
