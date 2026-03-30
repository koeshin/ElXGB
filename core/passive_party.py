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
        self.global_feature_slots = []
        self.total_feature_count = 0
        self.buckets = {}
        self.feature_bins = {}
        self.local_lookup_table = {}

    def set_data(self, X: np.ndarray, feature_names: list = None, global_feature_slots: list = None, total_feature_count: int = 0):
        self.X = np.array(X)
        self.feature_names = feature_names if feature_names else [f"f{i}" for i in range(self.X.shape[1])]
        self.global_feature_slots = global_feature_slots if global_feature_slots else list(range(1, self.X.shape[1] + 1))
        self.total_feature_count = total_feature_count or len(self.global_feature_slots)

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

    def find_best_split_plaintext(self, g_noisy_masked: np.ndarray, h_noisy_masked: np.ndarray, lambda_val: float = 1.0, gamma_val: float = 0.0):
        histograms = self.compute_plaintext_histograms(g_noisy_masked, h_noisy_masked)
        best = None

        for feat_idx, bins in histograms.items():
            total_g = sum(g for g, _ in bins)
            total_h = sum(h for _, h in bins)
            g_left, h_left = 0.0, 0.0

            for bin_idx in range(len(bins) - 1):
                g_i, h_i = bins[bin_idx]
                g_left += g_i
                h_left += h_i
                g_right = total_g - g_left
                h_right = total_h - h_left

                gain_left = (g_left ** 2) / (h_left + lambda_val) if (h_left + lambda_val) != 0 else 0.0
                gain_right = (g_right ** 2) / (h_right + lambda_val) if (h_right + lambda_val) != 0 else 0.0
                gain_total = (total_g ** 2) / (total_h + lambda_val) if (total_h + lambda_val) != 0 else 0.0
                gain = 0.5 * (gain_left + gain_right - gain_total) - gamma_val

                if best is None or gain > best["gain"]:
                    best = {"feature_idx": feat_idx, "bin_idx": bin_idx, "gain": float(gain)}

        return best

    def register_obfuscated_split(self, feat_idx: int, bin_idx: int, he_service=None) -> str:
        """Attribute & Direction Obfuscation"""
        record_id = str(uuid.uuid4())
        bins = self.feature_bins[feat_idx]
        actual_threshold = bins[bin_idx + 1]
        threshold_vector = None

        if he_service is not None:
            threshold_vector = [he_service.encrypt_scalar(self.global_feature_slots[feat_idx])]
            for slot in range(1, self.total_feature_count + 1):
                if slot == self.global_feature_slots[feat_idx]:
                    threshold_vector.append(he_service.encrypt_scalar(actual_threshold))
                else:
                    threshold_vector.append(he_service.encrypt_scalar(random.uniform(-1.0, 1.0)))
        
        self.local_lookup_table[record_id] = {
            "feature_local_idx": feat_idx,
            "threshold": actual_threshold,
            "global_feature_slot": self.global_feature_slots[feat_idx],
            "threshold_vector": threshold_vector
        }
        return record_id

    def split_dataset_mask(self, record_id: str, current_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        record = self.local_lookup_table[record_id]
        feat_idx = record["feature_local_idx"]
        threshold = record["threshold"]
        
        feature_array = self.X[:, feat_idx]
        left_cond = feature_array < threshold
        right_cond = ~left_cond
            
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
            
            feature_array = X_test[:, feat_idx]
            
            # 왼쪽 분기 조건 (기본값)
            left_cond = feature_array < threshold
                
            matrix[rid] = left_cond.tolist()
            
        return matrix
