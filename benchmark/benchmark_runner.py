import sys
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.elxgb_classifier import ELXGBClassifier
from core.plaintext_xgboost import PlaintextXGBoost

BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'benchmark_results')

def load_dataset(dataset_name):
    print(f"Loading dataset: {dataset_name}...")
    if dataset_name == 'bank_marketing':
        data = fetch_openml(data_id=1461, as_frame=True, parser='auto')
        df = data.frame
        target_col = data.target_names[0]
    elif dataset_name == 'credit_card':
        data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
        df = data.frame
        target_col = data.target_names[0]
    else:
        raise ValueError("Unsupported dataset")

    df = df.dropna()
    X_df = df.drop(columns=[target_col])
    y_series = df[target_col]

    cat_cols = X_df.select_dtypes(include=['category', 'object']).columns
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder()
        X_df[cat_cols] = encoder.fit_transform(X_df[cat_cols])
        
    le = LabelEncoder()
    y = le.fit_transform(y_series)

    X = X_df.to_numpy(dtype=np.float64)
    feature_names = X_df.columns.tolist()

    return X, y, feature_names

class BenchmarkRunner:
    def __init__(self, dataset_name, n_estimators=3, max_depth=3, bins=32, lr=1.0, dp_eps=10.0, num_parties=2):
        self.dataset_name = dataset_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bins = bins
        self.eps = 1.0 / bins
        self.lr = lr
        self.dp_eps = dp_eps
        self.num_parties = num_parties
        
        self.results_dir = os.path.join(
            BASE_RESULTS_DIR, 
            self.dataset_name, 
            f"depth{self.max_depth}_trees{self.n_estimators}_bins{self.bins}"
        )
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run(self, X, y, feature_names):
        print(f"\n[{self.dataset_name}] Trees: {self.n_estimators}, Depth: {self.max_depth}, Bins: {self.bins}, DP-eps: {self.dp_eps}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 특성 N분할 로직
        n_features = X.shape[1]
        chunk_size = max(1, n_features // self.num_parties)
        
        X_train_list = []
        X_test_list = []
        feat_names_list = []
        
        for i in range(self.num_parties):
            start_idx = i * chunk_size
            end_idx = n_features if i == self.num_parties - 1 else (i + 1) * chunk_size
            
            X_train_list.append(X_train[:, start_idx:end_idx])
            X_test_list.append(X_test[:, start_idx:end_idx])
            feat_names_list.append(feature_names[start_idx:end_idx])
            
        summary = {
            "config": {
                "dataset": self.dataset_name,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "bins": self.bins,
                "eps": self.eps,
                "learning_rate": self.lr,
                "dp_epsilon": self.dp_eps,
                "num_parties": self.num_parties,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "features": len(feature_names)
            },
            "results": {}
        }
        
        # ===== 1. XGBoost Library =====
        print(f"Dataset Split: Train={len(y_train)}, Test={len(y_test)}")
        print("[1/3] Training XGBoost Library...")
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.lr, tree_method='hist',
            max_bin=self.bins,
            reg_lambda=1.0, gamma=0.0, min_child_weight=0,
            base_score=0.5, random_state=42
        )
        xgb_model.fit(X_train_df, y_train)
        
        booster = xgb_model.get_booster()
        trees_json = booster.get_dump(dump_format='json', with_stats=True)
        parsed_trees = [json.loads(t) for t in trees_json]
        
        with open(os.path.join(self.results_dir, 'xgboost_library_tree.json'), 'w') as f:
            json.dump(parsed_trees, f, indent=4)
        
        acc_train_lib = accuracy_score(y_train, xgb_model.predict(X_train_df))
        acc_test_lib = accuracy_score(y_test, xgb_model.predict(X_test_df))
        print(f" -> XGB Lib Train: {acc_train_lib:.4f} | Test: {acc_test_lib:.4f}")
        summary["results"]["xgboost_library"] = {"train_acc": acc_train_lib, "test_acc": acc_test_lib}
        
        # ===== 2. Plaintext XGBoost (직접 구현) =====
        print("[2/3] Training Plaintext XGBoost (Our Implementation, No Crypto)...")
        plain_xgb = PlaintextXGBoost(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            eps=self.eps, learning_rate=self.lr
        )
        plain_xgb.fit(X_train, y_train, feature_names=feature_names)
        plain_xgb.export_model(os.path.join(self.results_dir, 'plaintext_xgboost_tree.json'))
        
        acc_train_plain = accuracy_score(y_train, plain_xgb.predict(X_train))
        acc_test_plain = accuracy_score(y_test, plain_xgb.predict(X_test))
        print(f" -> Plaintext Train: {acc_train_plain:.4f} | Test: {acc_test_plain:.4f}")
        summary["results"]["plaintext_xgboost"] = {"train_acc": acc_train_plain, "test_acc": acc_test_plain}
        
        # ===== 3. ELXGB (Federated + HE/DP) =====
        print("[3/3] Training ELXGB (Federated + HE/DP)...")
        elxgb = ELXGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            eps=self.eps, learning_rate=self.lr, dp_epsilon=self.dp_eps,
            num_passive_parties=self.num_parties
        )
        elxgb.fit(X_train_list, y_train, feat_names_list)
        elxgb.export_model(os.path.join(self.results_dir, 'elxgb_tree.json'))
        
        acc_train_elxgb = accuracy_score(y_train, elxgb.predict(X_train_list))
        acc_test_elxgb = accuracy_score(y_test, elxgb.predict(X_test_list))
        print(f" -> ELXGB Train: {acc_train_elxgb:.4f} | Test: {acc_test_elxgb:.4f}")
        summary["results"]["elxgb"] = {
            "train_acc": acc_train_elxgb, 
            "test_acc": acc_test_elxgb,
            "pure_train_time_sec": elxgb.total_pure_train_time,
            "total_comm_bytes": elxgb.total_comm_bytes,
            "total_comm_mb": elxgb.total_comm_bytes / 1024 / 1024
        }
        
        # JSON 요약 저장
        with open(os.path.join(self.results_dir, 'benchmark_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print("-" * 70)
        print(f"{'XGB Lib':<20} Test: {acc_test_lib:.4f}")
        print(f"{'Plaintext':<20} Test: {acc_test_plain:.4f}")
        print(f"{'ELXGB':<20} Test: {acc_test_elxgb:.4f}")
        print("-" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ELXGB Benchmark Runner')
    parser.add_argument('--dataset', type=str, default='bank_marketing', choices=['bank_marketing', 'credit_card'], help='Dataset name')
    parser.add_argument('--trees', type=int, default=3, help='Number of trees (n_estimators)')
    parser.add_argument('--depth', type=int, default=3, help='Max depth of each tree')
    parser.add_argument('--bins', type=int, default=32, help='Number of histogram bins')
    parser.add_argument('--dp_eps', type=float, default=10.0, help='DP epsilon')
    parser.add_argument('--parties', type=int, default=2, help='Number of passive parties')
    
    args = parser.parse_args()
    
    X, y, feature_names = load_dataset(args.dataset)
    
    runner = BenchmarkRunner(
        dataset_name=args.dataset,
        n_estimators=args.trees,
        max_depth=args.depth,
        bins=args.bins,
        dp_eps=args.dp_eps,
        num_parties=args.parties
    )
    runner.run(X, y, feature_names)
