from crypto.heservice import HEService
import numpy as np

class ActiveParty:
    """
    ELXGB - Active Party (PK)
    데이터 레이블(Target y)을 소유하고 있으며, 글로벌 트리를 구축하는 주체입니다.
    """
    def __init__(self, he_service: HEService):
        self.he_service = he_service
        self.scale_factor = 1.0  
        self.y_true = None
        self.y_pred = None

    def set_data(self, y_true: np.ndarray):
        """Active Party만의 고유 자산인 레이블(y) 수신"""
        self.y_true = y_true

    def initialize_predictions(self):
        """본격적인 학습(첫 번째 트리)을 시작하기 전, 예측 확률값(y_pred)을 통상 0.5로 초기화합니다."""
        if self.y_true is not None:
            self.y_pred = np.full(len(self.y_true), 0.5)

    def _compute_raw_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        """논문 Eq (5) 기반 1차 및 2차 미분 로컬 계산"""
        assert self.y_true is not None and self.y_pred is not None, "Data/Predictions not initialized."
        g = self.y_pred - self.y_true
        h = self.y_pred * (1.0 - self.y_pred)
        return g * self.scale_factor, h * self.scale_factor

    def calculate_optimal_split(
        self, 
        encrypted_histograms_per_party: dict, 
        lambda_val: float = 1.0, 
        gamma_val: float = 0.0
    ):
        """
        [HENS 전용] 논문 Eq (1) ~ (4) 분할(Gain) 탐색 알고리즘
        Passive Party(들)로부터 암호화된 히스토그램 빈들을 수신받아 복호화하고,
        각 특성(Feature)과 빈(Bin)을 기준으로 가장 높은 정보 획득량(Gain)을 내는 최적 분기점을 찾습니다.
        """
        max_gain = -np.inf
        best_split = None

        for party_id, feature_histograms in encrypted_histograms_per_party.items():
            for feature_idx, enc_bins in feature_histograms.items():
                decrypted_bins = []
                for enc_g, enc_h in enc_bins:
                    g_val = self.he_service.decrypt(enc_g)[0]
                    h_val = self.he_service.decrypt(enc_h)[0]
                    decrypted_bins.append((g_val, h_val))

                total_G = sum([b[0] for b in decrypted_bins])
                total_H = sum([b[1] for b in decrypted_bins])

                G_L, H_L = 0.0, 0.0
                for split_bin_idx in range(len(decrypted_bins) - 1):
                    g_i, h_i = decrypted_bins[split_bin_idx]
                    G_L += g_i
                    H_L += h_i
                    G_R = total_G - G_L
                    H_R = total_H - H_L

                    gain_L = (G_L ** 2) / (H_L + lambda_val)
                    gain_R = (G_R ** 2) / (H_R + lambda_val)
                    gain_Total = (total_G ** 2) / (total_H + lambda_val)
                    gain = 0.5 * (gain_L + gain_R - gain_Total) - gamma_val

                    if gain > max_gain:
                        max_gain = gain
                        best_split = (party_id, feature_idx, split_bin_idx)
                        
        return best_split, max_gain

    def compute_noisy_dp_gradients(self, dp_injector) -> tuple[np.ndarray, np.ndarray]:
        """
        [DPNS 전용] 순수 평문에 DP 노이즈만 입혀서 반환 (Algorithm 3)
        """
        g_raw, h_raw = self._compute_raw_gradients()
        g_noisy = dp_injector.inject_noise(g_raw.tolist())
        h_noisy = dp_injector.inject_noise(h_raw.tolist())
        return np.array(g_noisy), np.array(h_noisy)

    def find_global_optimal_split(self, party_best_splits: dict):
        """
        [DPNS 전용] 각 Passive Party들이 로컬에서 도출해온
        후보들 중 "Global 1등"을 선정합니다.
        
        :param party_best_splits: { "Party_1": (max_gain, feature_idx, split_bin_idx), ... }
        :return: (best_party_id, best_feature_idx, split_bin_idx), max_gain
        """
        global_max_gain = -np.inf
        best_split = None

        for p_name, (gain, f_idx, b_idx) in party_best_splits.items():
            if gain is not None and gain > global_max_gain:
                global_max_gain = gain
                best_split = (p_name, f_idx, b_idx)
                
        return best_split, global_max_gain
