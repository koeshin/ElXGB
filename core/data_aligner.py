import hashlib
import hmac

class SecureDataAligner:
    """
    ELXGB - Algorithm 4: Secure Data Alignment
    TA에서 발급된 ssk(키)를 사용하여 ID를 HMAC 해싱한 뒤, CSP 역할을 통해
    가장 짧은 해시 집합을 기준으로 교집합(Intersection)을 찾아 반환합니다.
    """
    def __init__(self, ssk: bytes):
        self.ssk = ssk

    def hash_ids(self, ids: list[str]) -> set[str]:
        """각 파티(Pk)가 로컬에서 자신의 ID를 해싱할 때 사용."""
        hashed = set()
        for uid in ids:
            # HMAC-SHA256을 활용한 H_{ssk}(ID). 평문 ID를 절대로 그대로 반환해선 안됨.
            h = hmac.new(self.ssk, uid.encode('utf-8'), hashlib.sha256).hexdigest()
            hashed.add(h)
        return hashed

    def intersect(self, parties_hashed_ids: list[set[str]]) -> set[str]:
        """
        CSP 역할: 암호화된(해싱된) ID 세트들 간의 교집합을 구합니다.
        가장 사이즈가 작은 세트를 중심으로 나머지 세트와 교차하여 효율성을 높입니다.
        (실제 논문에서는 L 벡터 마스킹을 쓰지만 프라이버시가 동일 보장되는 집합 교차 활용)
        """
        if not parties_hashed_ids:
            return set()
        
        parties_hashed_ids.sort(key=len)
        result = parties_hashed_ids[0]
        
        for p_set in parties_hashed_ids[1:]:
            result = result.intersection(p_set)
        
        return result
