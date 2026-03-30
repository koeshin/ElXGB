import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.elxgb_classifier import ELXGBClassifier


def _internal_nodes(node):
    if "leaf" in node:
        return []
    nodes = [node]
    for child in node["children"]:
        nodes.extend(_internal_nodes(child))
    return nodes


def test_fit_and_plaintext_predict_work_and_track_metrics():
    x_party_1 = np.array([[0.1], [0.2], [0.8], [0.9], [0.15], [0.85]], dtype=np.float64)
    x_party_2 = np.array([[1.0], [0.9], [0.2], [0.1], [0.95], [0.05]], dtype=np.float64)
    y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)

    model = ELXGBClassifier(
        n_estimators=2,
        max_depth=2,
        eps=0.5,
        learning_rate=0.5,
        dp_epsilon=20.0,
        num_passive_parties=2,
    )
    model.fit([x_party_1, x_party_2], y, [["a1"], ["b1"]])

    predictions = model.predict([x_party_1, x_party_2])

    assert predictions.shape == y.shape
    assert model.total_pure_train_time >= 0.0
    assert model.total_comm_bytes > 0
    assert all("split" not in node and "split_condition" not in node for tree in model.trees for node in _internal_nodes(tree))

