import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crypto.heservice import HEService
from core.passive_party import PassiveParty


def test_register_obfuscated_split_creates_hidden_threshold_vector():
    he = HEService()
    party = PassiveParty(eps=0.5)
    x = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float64)
    party.set_data(x, ["feat"], global_feature_slots=[1], total_feature_count=1)
    party.generate_global_buckets()

    record_id = party.register_obfuscated_split(0, 0, he_service=he)
    record = party.local_lookup_table[record_id]

    assert record["threshold_vector"] is not None
    assert len(record["threshold_vector"]) == 2
    assert int(round(he.decrypt_scalar(record["threshold_vector"][0]))) == 1
    assert abs(he.decrypt_scalar(record["threshold_vector"][1]) - record["threshold"]) < 1e-6

