import sys
import os
import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.elxgb_classifier import ELXGBClassifier

def run_secure_offline_inference_demo():
    print("=== [ELXGB] Secure Offline Inference Demo ===")
    
    # 데이터 준비
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    split_idx = 15
    X_train_A, X_train_B = X_train[:, :split_idx], X_train[:, split_idx:]
    X_test_A, X_test_B = X_test[:, :split_idx], X_test[:, split_idx:]
    feat_A, feat_B = feature_names[:split_idx], feature_names[split_idx:]
    
    X_train_list = [X_train_A, X_train_B]
    X_test_list = [X_test_A, X_test_B]
    feat_names_list = [feat_A, feat_B]
    
    print("\n[1] Training ELXGB Model...")
    # DP-eps 10.0으로 완화하여 적당히 수렴 & 속도 확보
    max_bin = 256 # Assuming a default max_bin value for the new ELXGBClassifier parameter
    elxgb = ELXGBClassifier(n_estimators=3, max_depth=3, eps=1.0/max_bin, learning_rate=1.0, dp_epsilon=10.0, num_passive_parties=2)
    elxgb.fit(X_train_list, y_train, feat_names_list)
    
    # ---------------------------------------------------------
    # 단계 1: 일반 평문(On-the-fly) 방식 평가 (Baseline Speed)
    # ---------------------------------------------------------
    print("\n[2] Testing Plaintext(On-the-fly) Inference...")
    start_time = time.time()
    y_pred_plain = elxgb.predict(X_test_list)
    plain_time = time.time() - start_time
    plain_acc = accuracy_score(y_test, y_pred_plain)
    print(f" -> Accuracy: {plain_acc:.4f} | Time: {plain_time:.4f} sec")


    # ---------------------------------------------------------
    # 단계 2: Secure Offline Inference
    # ---------------------------------------------------------
    print("\n[3] Testing Secure Offline Inference...")
    
    # 1. 글로벌 모델에서 난독화 식별자(record_ids) 추출
    record_ids = elxgb.get_all_record_ids()
    print(f" -> Extracted {len(record_ids)} unique obfuscated node records from Global Model.")
    
    # 2. 파티별로 오프라인 이진 매트릭스 생성 (각 파티는 자신의 노드만 응답하고 나머지는 무시함)
    # Passive Party들에게 글로벌 모델의 난독화 노드 ID 리스트를 전송하고, 매트릭스를 반환받음
    mat_start_time = time.time()
    inference_matrix_A = elxgb.passive_parties["Party_1"].generate_inference_matrix(record_ids, X_test_list[0])
    inference_matrix_B = elxgb.passive_parties["Party_2"].generate_inference_matrix(record_ids, X_test_list[1])
    mat_gen_time = time.time() - mat_start_time
    
    inference_matrices = [inference_matrix_A, inference_matrix_B]
    print(f" -> Offline Matrices generated purely locally (Time: {mat_gen_time:.4f} sec).")
    
    # 3. 글로벌 분류기(또는 Active Party)가 제공받은 매트릭스를 가지고 오프라인 추론 수행
    eval_start_time = time.time()
    # 파티 A, B의 매트릭스 제출
    y_pred_offline = elxgb.predict_offline(inference_matrices, num_samples=len(y_test))
    eval_time = time.time() - eval_start_time
    offline_acc = accuracy_score(y_test, y_pred_offline)
    
    print(f" -> Accuracy: {offline_acc:.4f} | Offline Eval Time: {eval_time:.4f} sec")
    
    # 검증: 평문 추론과 매트릭스 기반 추론이 완전히 일치하는지 확인!
    is_identical = np.array_equal(y_pred_plain, y_pred_offline)
    print(f"\n[Conclusion] Are Plaintext and Offline Inference identical? {'YES' if is_identical else 'NO'}")
    print("===========================================")

if __name__ == "__main__":
    run_secure_offline_inference_demo()
