import numpy as np
import uuid
import random

class PassiveParty:
    """
    ELXGB - Passive Party (Pk)
    데이터의 로컬 특성(Feature X)들만 보유하며, 레이블(y)은 절대로 열람할 수 없는 기관입니다.
    Quantile Sketch 방식으로 특성값을 히스토그램 빈(Bin)으로 매핑합니다.
    np.digitize를 사용하여 경계값 누락 없는 안전한 빈 할당을 보장합니다.
    """
    def __init__(self, eps: float = 0.1):
        self.eps = eps
        self.X = None
        self.feature_names = None
        self.buckets = {}
        self.feature_bins = {}
        self.local_lookup_table = {}

    def set_data(self, X: np.ndarray, feature_names: list = None):
        self.X = np.array(X)
        self.feature_names = feature_names if feature_names else [f"f{i}" for i in range(self.X.shape[1])]

    def generate_global_buckets(self):
        """
        np.quantile + np.digitize 기반 안전한 Quantile Sketch 버킷 생성.
        모든 샘플이 반드시 하나의 빈에 할당됩니다.
        """
        assert self.X is not None, "Feature data (X) is not initialized."
        n_samples, m_features = self.X.shape
        num_buckets_target = max(1, int(1.0 / self.eps))
        
        self.buckets = {}
        self.feature_bins = {}

        for attr_idx in range(m_features):
            feature_values = self.X[:, attr_idx]
            
            quantiles = np.linspace(0, 1, num_buckets_target + 1)
            bins = np.quantile(feature_values, quantiles)
            bins = np.unique(bins)
            
            self.feature_bins[attr_idx] = bins
            num_actual_buckets = len(bins) - 1
            
            # np.digitize: bins[1:]을 경계로 사용하면 0 ~ num_actual_buckets-1 인덱스 반환
            bin_indices = np.digitize(feature_values, bins[1:], right=False)
            # 상한 클리핑: 마지막 경계값과 같은 값은 마지막 빈에 할당
            bin_indices = np.clip(bin_indices, 0, num_actual_buckets - 1)
            
            bucket_vectors = []
            for b_idx in range(num_actual_buckets):
                mask = (bin_indices == b_idx).astype(int)
                bucket_vectors.append(mask)
                
            self.buckets[attr_idx] = bucket_vectors

    def compute_encrypted_histograms(self, enc_g, enc_h, current_mask: np.ndarray):
        """
        HENS: 암호화된 전체 그래디언트 벡터와 '해당 노드의 마스크(current_mask)'를 
        고려하여 히스토그램 도출
        """
        encrypted_histograms = {}
        
        # 디버깅 출력 플래그 (트리당 최초 1회만 출력하도록)
        debug_printed = True
        
        for attr_idx, vectors in self.buckets.items():
            bin_histograms = []
            for I_vec in vectors:
                # 노드에 도달한 샘플과 빈에 포함된 샘플의 교집합
                valid_mask = (I_vec & current_mask).astype(float)
                

                valid_mask += 1e-10  ## 히스토그램의 값이 모두 0이면 dot-product가 안되서 작은 값 추가
                
                valid_mask = valid_mask.tolist()
                sum_g = enc_g.dot(valid_mask)
                sum_h = enc_h.dot(valid_mask)
                # print('======== dot done ========')
                bin_histograms.append((sum_g, sum_h))
            encrypted_histograms[attr_idx] = bin_histograms
        return encrypted_histograms

    def compute_plaintext_histograms(self, g_noisy_masked: np.ndarray, h_noisy_masked: np.ndarray) -> dict:
        """
        DPNS: 평문 그래디언트 고속 히스토그램 연산.
        입력되는 g_noisy_masked는 이미 current_mask 처리가 완료된 상태.
        """
        histograms = {}
        for attr_idx, vectors in self.buckets.items():
            bin_histograms = []
            for I_vec in vectors:
                # I_vec (Int Array 0/1) 와 마스킹된 gradient 스칼라곱
                sum_g = I_vec.dot(g_noisy_masked)
                sum_h = I_vec.dot(h_noisy_masked)
                bin_histograms.append((float(sum_g), float(sum_h)))
            histograms[attr_idx] = bin_histograms
        return histograms

    def register_obfuscated_split(self, feat_idx: int, bin_idx: int) -> str:
        """Attribute & Direction Obfuscation"""
        record_id = str(uuid.uuid4())
        bins = self.feature_bins[feat_idx]
        actual_threshold = bins[bin_idx + 1]
        is_flipped = random.choice([True, False])
        
        self.local_lookup_table[record_id] = {
            "feature_local_idx": feat_idx,
            "threshold": actual_threshold,
            "is_flipped": is_flipped
        }
        return record_id

    def split_dataset_mask(self, record_id: str, current_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        record = self.local_lookup_table[record_id]
        feat_idx = record["feature_local_idx"]
        threshold = record["threshold"]
        is_flipped = record["is_flipped"]
        
        feature_array = self.X[:, feat_idx]
        left_cond = feature_array < threshold
        right_cond = ~left_cond
        
        if is_flipped:
            left_cond, right_cond = right_cond, left_cond
            
        left_mask = current_mask & left_cond
        right_mask = current_mask & right_cond
        return left_mask, right_mask

    def generate_inference_matrix(self, record_ids: list, X_test: np.ndarray) -> dict:
        """
        [Secure Offline Inference] 
        각 테스트 샘플이 특정 record_id에서 왼쪽(True)으로 갈지 오른쪽(False)으로 갈지를 
        사전에 모두 판별하여 이진 매트릭스로 생성합니다.
        """
        matrix = {}
        for rid in record_ids:
            if rid not in self.local_lookup_table:
                continue
                
            record = self.local_lookup_table[rid]
            feat_idx = record["feature_local_idx"]
            threshold = record["threshold"]
            is_flipped = record["is_flipped"]
            
            feature_array = X_test[:, feat_idx]
            
            # 왼쪽 분기 조건 (기본값)
            left_cond = feature_array < threshold
            # 방향 난독화(Direction Obfuscation)가 적용되었으면 뒤집음
            if is_flipped:
                left_cond = ~left_cond
                
            matrix[rid] = left_cond.tolist()
            
        return matrix

    def calculate_local_optimal_split_plaintext(self, g_noisy, h_noisy, current_mask, lambda_val=1.0, gamma_val=0.0):
        """
        (DPNS 전용) Active Party가 트리 시작 전 딱 1회 DP 노이즈를 주입해둔
        가짜 오차(g_noisy, h_noisy)를 수신받아, 로컬에서 히스토그램을 묶고
        자체적으로 Gain을 스캔하여 "우리 파티의 1등 후보" 하나만 도출합니다.
        
        [핵심] 여기서는 추가적인 노이즈 주입이 절대 없습니다!
        노이즈는 Active Party가 이미 전체 오차 배열에 1회 씌운 상태입니다.
        통신량 O(1) 달성의 핵심 로직.
        """
        # 1. current_mask로 해당 노드에 속하는 샘플만 필터링
        g_masked = g_noisy * current_mask
        h_masked = h_noisy * current_mask
        
        # 2. 로컬 히스토그램 집계 (이미 노이즈가 섞인 평문 기반)
        hists = self.compute_plaintext_histograms(g_masked, h_masked)
        
        # 3. Gain 스캔 (추가 노이즈 없이 그대로 사용)
        max_gain = -float('inf')
        local_best_feat_idx = None
        local_best_bin_idx = None
        
        for feature_idx, bins in hists.items():
            total_G = sum(b[0] for b in bins)
            total_H = sum(b[1] for b in bins)
            
            G_L, H_L = 0.0, 0.0
            for split_bin_idx in range(len(bins) - 1):
                g_i, h_i = bins[split_bin_idx]
                G_L += g_i
                H_L += h_i
                
                G_R = total_G - G_L
                H_R = total_H - H_L
                
                gain_L = (G_L ** 2) / (H_L + lambda_val) if (H_L + lambda_val) != 0 else 0
                gain_R = (G_R ** 2) / (H_R + lambda_val) if (H_R + lambda_val) != 0 else 0
                gain_Total = (total_G ** 2) / (total_H + lambda_val) if (total_H + lambda_val) != 0 else 0
                
                gain = 0.5 * (gain_L + gain_R - gain_Total) - gamma_val
                
                if gain > max_gain:
                    max_gain = gain
                    local_best_feat_idx = feature_idx
                    local_best_bin_idx = split_bin_idx
                    
        # 무거운 전체 히스토그램 대신 1등 메타데이터 튜플 1개만 리턴
        return max_gain, local_best_feat_idx, local_best_bin_idx
