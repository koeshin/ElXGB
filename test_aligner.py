from data_aligner import SecureDataAligner

def test_data_alignment():
    print("Testing ELXGB Algorithm 4: Secure Data Alignment...")
    
    # 1. TA generates a secret key (ssk)
    ssk = b"super_secret_elxgb_key"
    aligner = SecureDataAligner(ssk)
    
    # 2. 3개의 파티가 서로 다른 ID 세트를 가짐 (1과 2가 공통)
    party_1_ids = ["ID_1", "ID_2", "ID_3", "ID_4"]
    party_2_ids = ["ID_0", "ID_1", "ID_2", "ID_5"]
    party_3_ids = ["ID_1", "ID_2", "ID_7"]
    
    # 3. [Local Phase] 각 파티는 데이터를 해싱해서 보냄 (절대 원문을 보내지 않음)
    p1_hashed = aligner.hash_ids(party_1_ids)
    p2_hashed = aligner.hash_ids(party_2_ids)
    p3_hashed = aligner.hash_ids(party_3_ids)
    
    # 4. [CSP Phase] 해시값들만 이용해서 교집합 반환
    aligned_hashes = aligner.intersect([p1_hashed, p2_hashed, p3_hashed])
    
    print(f"Intersection Size: {len(aligned_hashes)}")
    assert len(aligned_hashes) == 2, "Intersection logic failed!"
    
    # 5. [Validation] 원문 ID "ID_1"과 "ID_2"의 해시값이 정확히 포함되었는지 검증
    expected_hash_1 = list(aligner.hash_ids(["ID_1"]))[0]
    expected_hash_2 = list(aligner.hash_ids(["ID_2"]))[0]
    
    assert expected_hash_1 in aligned_hashes
    assert expected_hash_2 in aligned_hashes
    print("Test Passed: Only common hashed IDs securely extracted without revealing raw IDs.")

if __name__ == "__main__":
    test_data_alignment()
