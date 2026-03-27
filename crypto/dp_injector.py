import numpy as np
from diffprivlib.mechanisms import GaussianAnalytic

class DPNoiseInjector:
    """
    ELXGB - Algorithm 3: Centralized Differential Privacy Algorithm
    신뢰할 수 있는 IBM diffprivlib의 Gaussian Mechanism으로 전면 교체하여
    수학적 무결성 및 엄격한 (epsilon, delta)-DP를 보장합니다.
    """
    def __init__(self, epsilon: float, delta: float, clip_c: float):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_c = clip_c
        
        # Sensitivity = 2 * C
        self.sensitivity = 2 * self.clip_c
        
        # IBM Diffprivlib GaussianAnalytic 모듈 초기화 (epsilon > 1.0 지원)
        self.mech = GaussianAnalytic(
            epsilon=self.epsilon, 
            delta=self.delta, 
            sensitivity=self.sensitivity
        )

    def inject_noise(self, gradients: list[float]) -> list[float]:
        """그래디언트 배열에 검증된 IBM 라이브러리 가우시안 DP 노이즈 주입"""
        clipped = np.clip(gradients, -self.clip_c, self.clip_c)
        return [float(self.mech.randomise(g)) for g in clipped]
