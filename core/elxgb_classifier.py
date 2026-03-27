import numpy as np
import json
from crypto.heservice import HEService
from crypto.dp_injector import DPNoiseInjector
from core.active_party import ActiveParty
from core.passive_party import PassiveParty

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NodeIdCounter:
    def __init__(self):
        self.id = 0
    def get(self):
        curr = self.id
        self.id += 1
        return curr

class ELXGBClassifier:
    """
    ELXGB (Federated Learning) 메인 분류기 시스템
    N개의 Passive Party를 동적으로 지원하며, 트리 구축 시
    암호화 및 DP 노이즈 주입을 1회만 수행하는 고도의 최적화가 적용되었습니다.
    """
    def __init__(self, n_estimators=3, max_depth=5, eps=0.125, learning_rate=1.0, 
                 lambda_val=1.0, gamma_val=0.0, dp_epsilon=10.0, dp_clip_c=2.0, num_passive_parties=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eps = eps
        self.lr = learning_rate
        self.lambda_val = lambda_val
        self.gamma_val = gamma_val
        self.trees = []
        self.base_score = 0.5
        
        self.he_svc = HEService()
        self.active_party = ActiveParty(self.he_svc)
        
        # N개의 패시브 파티 동적 생성
        self.passive_parties = {
            f"Party_{i+1}": PassiveParty(eps=self.eps) 
            for i in range(num_passive_parties)
        }
        
        self.dp_injector = DPNoiseInjector(epsilon=dp_epsilon, delta=1e-5, clip_c=dp_clip_c)

    def fit(self, X_list: list, y: np.ndarray, feat_names_list: list = None):
        """
        N개의 Passive Party 데이터 리스트를 입력받아 분산 훈련 수행
        X_list: [Party_1_Data, Party_2_Data, ...]
        """
        assert len(X_list) == len(self.passive_parties), "Mismatch between X_list length and number of passive parties"
        
        if feat_names_list is None:
            feat_names_list = [None] * len(X_list)
            
        self.active_party.set_data(y)
        
        for (p_name, p_obj), X_data, feat_names in zip(self.passive_parties.items(), X_list, feat_names_list):
            p_obj.set_data(X_data, feat_names)
            p_obj.generate_global_buckets()
        
        current_margins = np.zeros(len(y))
        
        for t in range(self.n_estimators):
            method_str = "HENS (TenSEAL Homomorphic Encryption)" if t == 0 else "DPNS (IBM Differential Privacy)"
            print(f"[ELXGB] Building Ensemble Tree {t+1}/{self.n_estimators} via {method_str}...")
            
            self.active_party.y_pred = sigmoid(current_margins)
            
            # ==========================================================
            # 🌟 [최적화] Pre-preparation: 트리당 딱 1번만 수행!
            # ==========================================================
            g_raw, h_raw = self.active_party._compute_raw_gradients()
            
            enc_g, enc_h = None, None
            g_noisy, h_noisy = None, None
            
            if t == 0:
                # 첫 트리는 HENS를 위해 단 한 번만 암호화
                enc_g = self.he_svc.encrypt(g_raw.tolist())
                enc_h = self.he_svc.encrypt(h_raw.tolist())
            else:
                # 두 번째부터는 전체 벡터에 노이즈를 단 한 번만 주입
                g_noisy, h_noisy = self.active_party.compute_noisy_dp_gradients(self.dp_injector)
            # ==========================================================
            
            node_counter = NodeIdCounter()
            initial_mask = np.ones(len(y), dtype=bool)
            
            tree_struct = self._build_tree_recursive(
                current_mask=initial_mask, 
                depth=0, 
                node_counter=node_counter, 
                tree_idx=t,
                g_raw=g_raw, h_raw=h_raw,
                enc_g=enc_g, enc_h=enc_h,
                g_noisy=g_noisy, h_noisy=h_noisy
            )
            self.trees.append(tree_struct)
            
            # 다음 트리를 위해 마진 업데이트
            tree_margins = np.array([self._predict_single_tree(tree_struct, x_list) for x_list in zip(*X_list)])
            current_margins += self.lr * tree_margins

    def _build_tree_recursive(self, current_mask, depth, node_counter, tree_idx, 
                              g_raw, h_raw, enc_g, enc_h, g_noisy, h_noisy):
        node_id = node_counter.get()
        print('current tree depth:',depth)
        # 원본 그래디언트로 노드 가중치 및 종료조건(Gain) 평가
        G_total = np.sum(g_raw[current_mask])
        H_total = np.sum(h_raw[current_mask])
        
        if depth >= self.max_depth or np.sum(current_mask) < 2 or H_total == 0:
            leaf_weight = -G_total / (H_total + self.lambda_val)
            return {"nodeid": node_id, "leaf": float(leaf_weight)}
            
        # HENS (첫 번째 트리)
        if tree_idx == 0:
            histograms = {}
            for p_name, p_obj in self.passive_parties.items():
                # 노드 분할 평가를 위해 current_mask를 함께 넘김
                histograms[p_name] = p_obj.compute_encrypted_histograms(enc_g, enc_h, current_mask)
            
            best_split, max_gain = self.active_party.calculate_optimal_split(
                histograms, lambda_val=self.lambda_val, gamma_val=self.gamma_val
            )
        # DPNS (두 번째 트리 이후)
        else:
            g_masked = g_noisy * current_mask
            h_masked = h_noisy * current_mask
            
            histograms = {}
            for p_name, p_obj in self.passive_parties.items():
                histograms[p_name] = p_obj.compute_plaintext_histograms(g_masked, h_masked)
            
            best_split, max_gain = self.active_party.calculate_optimal_split_plaintext(
                histograms, lambda_val=self.lambda_val, gamma_val=self.gamma_val
            )
        
        if best_split is None or max_gain <= 0:
            leaf_weight = -G_total / (H_total + self.lambda_val)
            return {"nodeid": node_id, "leaf": float(leaf_weight)}
            
        best_party, best_feat_idx, best_bin_idx = best_split
        party_obj = self.passive_parties[best_party]
        feat_name = party_obj.feature_names[best_feat_idx]
        
        record_id = party_obj.register_obfuscated_split(best_feat_idx, best_bin_idx)
        rec = party_obj.local_lookup_table[record_id]
        
        left_mask, right_mask = party_obj.split_dataset_mask(record_id, current_mask)
        
        left_child = self._build_tree_recursive(left_mask, depth + 1, node_counter, tree_idx, 
                                                g_raw, h_raw, enc_g, enc_h, g_noisy, h_noisy)
        right_child = self._build_tree_recursive(right_mask, depth + 1, node_counter, tree_idx, 
                                                 g_raw, h_raw, enc_g, enc_h, g_noisy, h_noisy)
        
        return {
            "nodeid": node_id,
            "depth": depth,
            "split": str(feat_name),
            "split_condition": float(rec["threshold"]),
            "party": best_party,
            "record_id": record_id,
            "gain": float(max_gain),
            "yes": left_child["nodeid"],
            "no": right_child["nodeid"],
            "children": [left_child, right_child]
        }

    def _predict_single_tree(self, node, x_list):
        if "leaf" in node:
            return node["leaf"]
            
        party_idx = int(node["party"].split("_")[1]) - 1
        x_target = x_list[party_idx]
        
        party_obj = self.passive_parties[node["party"]]
        rec = party_obj.local_lookup_table[node["record_id"]]
        
        val = x_target[rec["feature_local_idx"]]
        
        go_left = val < rec["threshold"]
        if rec["is_flipped"]:
            go_left = not go_left
            
        if go_left:
            return self._predict_single_tree(node["children"][0], x_list)
        else:
            return self._predict_single_tree(node["children"][1], x_list)

    def predict_proba(self, X_list: list):
        margins = np.zeros(len(X_list[0]))
        for tree in self.trees:
            tree_margins = np.array([self._predict_single_tree(tree, x_list) for x_list in zip(*X_list)])
            margins += self.lr * tree_margins
        return sigmoid(margins)

    def predict(self, X_list: list):
        probs = self.predict_proba(X_list)
        return (probs > 0.5).astype(int)

    def export_model(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.trees, f, indent=4)

    def get_all_record_ids(self) -> list:
        rids = set()
        def _traverse(node):
            if "leaf" in node: return
            rids.add(node["record_id"])
            _traverse(node["children"][0])
            _traverse(node["children"][1])
        for tree in self.trees:
            _traverse(tree)
        return list(rids)

    def predict_offline_proba(self, inference_matrices: list, num_samples: int) -> np.ndarray:
        global_matrix = {}
        for mat in inference_matrices:
            global_matrix.update(mat)
            
        margins = np.zeros(num_samples)
        for tree in self.trees:
            margins += self.lr * self._predict_tree_offline_vectorized(tree, global_matrix, num_samples)
        return sigmoid(margins)

    def _predict_tree_offline_vectorized(self, tree, global_matrix, num_samples) -> np.ndarray:
        out = np.zeros(num_samples)
        
        def _traverse(node, indices):
            if len(indices) == 0:
                return
            if "leaf" in node:
                out[indices] = node["leaf"]
                return
            rid = node["record_id"]
            decisions = np.array(global_matrix[rid])[indices]
            left_indices = indices[decisions]
            right_indices = indices[~decisions]
            
            _traverse(node["children"][0], left_indices)
            _traverse(node["children"][1], right_indices)
            
        _traverse(tree, np.arange(num_samples))
        return out

    def predict_offline(self, inference_matrices: list, num_samples: int) -> np.ndarray:
        probs = self.predict_offline_proba(inference_matrices, num_samples)
        return (probs > 0.5).astype(int)
