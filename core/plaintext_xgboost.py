"""
PlaintextXGBoost: 암호화/DP 전혀 없는 순수 평문 XGBoost 직접 구현
ELXGB와 동일한 Quantile Sketch 빈닝 + np.digitize 안전 할당 사용.
"""

import numpy as np
import json


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PlaintextXGBoost:
    def __init__(self, n_estimators=3, max_depth=4, eps=0.125, learning_rate=1.0, lambda_val=1.0, gamma_val=0.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eps = eps
        self.lr = learning_rate
        self.lambda_val = lambda_val
        self.gamma_val = gamma_val
        self.trees = []

    def _generate_bins(self, X):
        """각 피처에 대해 np.quantile 기반 분할점 생성"""
        num_buckets = max(1, int(1.0 / self.eps))
        quantiles = np.linspace(0, 1, num_buckets + 1)
        self.feature_bins = {}
        for col in range(X.shape[1]):
            bins = np.quantile(X[:, col], quantiles)
            bins = np.unique(bins)
            self.feature_bins[col] = bins

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])]
        self.X_train = X
        self._generate_bins(X)
        current_margins = np.zeros(len(y))

        for t in range(self.n_estimators):
            print(f"[PlaintextXGB] Building Tree {t+1}/{self.n_estimators}...")
            y_pred = sigmoid(current_margins)
            g = y_pred - y
            h = y_pred * (1.0 - y_pred)

            mask = np.ones(len(y), dtype=bool)
            tree = self._build_tree(X, g, h, mask, depth=0)
            self.trees.append(tree)

            margins = np.array([self._predict_node(tree, x) for x in X])
            current_margins += self.lr * margins

    def _build_tree(self, X, g, h, mask, depth):
        G = np.sum(g[mask])
        H = np.sum(h[mask])

        if depth >= self.max_depth or np.sum(mask) < 2 or H == 0:
            return {"leaf": float(-G / (H + self.lambda_val))}

        best_gain = -np.inf
        best_feat = None
        best_threshold = None

        for col, bins in self.feature_bins.items():
            num_bins = len(bins) - 1
            bin_g = np.zeros(num_bins)
            bin_h = np.zeros(num_bins)
            vals = X[:, col]

            # np.digitize 안전한 빈 할당 (버그 수정)
            bin_indices = np.digitize(vals, bins[1:], right=False)
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)

            for b_idx in range(num_bins):
                in_bin = mask & (bin_indices == b_idx)
                bin_g[b_idx] = np.sum(g[in_bin])
                bin_h[b_idx] = np.sum(h[in_bin])

            G_L, H_L = 0.0, 0.0
            for b_idx in range(num_bins - 1):
                G_L += bin_g[b_idx]
                H_L += bin_h[b_idx]
                G_R = G - G_L
                H_R = H - H_L

                gain_L = (G_L ** 2) / (H_L + self.lambda_val) if (H_L + self.lambda_val) != 0 else 0
                gain_R = (G_R ** 2) / (H_R + self.lambda_val) if (H_R + self.lambda_val) != 0 else 0
                gain_T = (G ** 2) / (H + self.lambda_val) if (H + self.lambda_val) != 0 else 0
                gain = 0.5 * (gain_L + gain_R - gain_T) - self.gamma_val

                if gain > best_gain:
                    best_gain = gain
                    best_feat = col
                    best_threshold = float(bins[b_idx + 1])

        if best_feat is None or best_gain <= 0:
            return {"leaf": float(-G / (H + self.lambda_val))}

        left_mask = mask & (X[:, best_feat] < best_threshold)
        right_mask = mask & (X[:, best_feat] >= best_threshold)

        left_child = self._build_tree(X, g, h, left_mask, depth + 1)
        right_child = self._build_tree(X, g, h, right_mask, depth + 1)

        return {
            "depth": depth,
            "split": self.feature_names[best_feat],
            "split_condition": best_threshold,
            "gain": float(best_gain),
            "children": [left_child, right_child]
        }

    def _predict_node(self, node, x):
        if "leaf" in node:
            return node["leaf"]
        feat_idx = self.feature_names.index(node["split"])
        if x[feat_idx] < node["split_condition"]:
            return self._predict_node(node["children"][0], x)
        else:
            return self._predict_node(node["children"][1], x)

    def predict_proba(self, X):
        margins = np.zeros(len(X))
        for tree in self.trees:
            margins += self.lr * np.array([self._predict_node(tree, x) for x in X])
        return sigmoid(margins)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def export_model(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.trees, f, indent=4)
