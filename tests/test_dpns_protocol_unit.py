import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.passive_party import PassiveParty


def test_dpns_local_best_split_returns_only_best_candidate():
    party = PassiveParty(eps=0.5)
    x = np.array(
        [
            [0.1, 10.0],
            [0.2, 10.1],
            [0.8, 10.2],
            [0.9, 10.3],
        ],
        dtype=np.float64,
    )
    party.set_data(x, ["good", "weak"], global_feature_slots=[1, 2], total_feature_count=2)
    party.generate_global_buckets()

    g = np.array([0.4, 0.3, -0.4, -0.3], dtype=np.float64)
    h = np.full(4, 0.25, dtype=np.float64)

    candidate = party.find_best_split_plaintext(g, h, lambda_val=1.0, gamma_val=0.0)

    assert candidate is not None
    assert set(candidate.keys()) == {"feature_idx", "bin_idx", "gain"}
    assert candidate["feature_idx"] == 0
