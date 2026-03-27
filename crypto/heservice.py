from phe import paillier

class PaillierVector:
    """
    ELXGB 논문 원본의 Paillier 암호화를 완벽히 재현하기 위한 래퍼 클래스입니다.
    다수의 단일 스칼라 암호문들을 리스트로 묶어 배열(벡터)처럼 다루게 해줍니다.
    """
    def __init__(self, encrypted_list):
        self.data = encrypted_list

    def dot(self, plain_vector: list):
        """
        [논문 수식 완벽 재현]
        [[sum(g_i * I_i)]] = [[g_1]]^{I_1} * [[g_2]]^{I_2} * ... * [[g_N]]^{I_N}
        
        - Step A (스칼라 곱): 암호문에 마스크 값(0 or 1)을 파이썬 스칼라 곱(*)으로 수행하면 
          내부 수학적으로 Paillier 거듭제곱 연산이 이루어집니다.
        - Step B (가산 준동형성): 파이썬 덧셈 기호(+)를 수행하면 
          내부 수학적으로 Paillier 암호문 간의 곱 연산이 이루어집니다.
        """
        total = 0
        
        for enc_val, mask_val in zip(self.data, plain_vector):
            # 1. 마스크 값을 확실한 0 또는 1 정수로 변환
            mask_int = 1 if mask_val > 0.5 else 0
            
            # Step A: 스칼라 곱 (내부적으로 [[g_i]]^{I_i} 거듭제곱 처리됨)
            # 만약 mask_int가 0이면 결과는 평문 '0'을 의미하는 암호문이 반환됨
            masked_enc_val = enc_val * mask_int
            
            # Step B: 암호문 간의 누적 곱 (파이썬 phe에서는 + 로 오버로딩됨)
            total = total + masked_enc_val
            
        return total

class HEService:
    """
    ELXGB - Homomorphic Encryption Service (Paillier 기반)
    python-paillier (phe) 라이브러리를 사용하여 ELXGB 원본 논문의 수식을 재현합니다.
    """
    def __init__(self):
        print("[HEService] 초기화: Paillier 공개키/비밀키 쌍 생성 중... (조금 오래 걸릴 수 있습니다)")
        # 속도 최적화를 위해 사용자 요청에 따라 512 비트 키 길이 적용 (보안성은 낮아지나 연산 대폭 향상)
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=512)
        print("[HEService] Paillier 512비트 스몰 키 생성 완료.")
        
    def encrypt(self, vector: list[float]) -> PaillierVector:
        """
        넘어온 리스트의 스칼라 값들을 For문을 돌며 하나하나 모두 암호화합니다.
        데이터셋이 커지면 (수만 건) 시간이 엄청나게 소요됩니다.
        """
        # print(f"[HEService] {len(vector)}개의 요소를 하나씩 Paillier 암호화 중...")
        encrypted_list = [self.public_key.encrypt(float(x)) for x in vector]
        return PaillierVector(encrypted_list)

    def decrypt(self, enc_item) -> list[float]:
        """
        [Active Party 전용] 암호화된 벡터(히스토그램 등)를 평문으로 복호화합니다.
        기존 TenSEAL 구조와의 100% 호환성을 위해 리스트(크기 1) 형태로 감싸서 반환합니다.
        """
        if isinstance(enc_item, (int, float)):
            # 아무런 빈에 들어가지 않아 0이 반환되었을 경우
            return [float(enc_item)]
            
        value = self.private_key.decrypt(enc_item)
        return [float(value)]
    
    def add(self, enc_vec1, enc_vec2):
        """암호 상태에서의 덧셈 (Paillier Additive Homomorphism 수식 적용)"""
        return enc_vec1 + enc_vec2
