import tenseal as ts

class HEService:
    """
    ELXGB - Homomorphic Encryption Service (HENS 기반)
    논문(Paillier) 구조를 최신 TenSEAL(CKKS)로 구조화하여,
    연합학습에서 Active Party와 Passive Party 간의 암호화된
    Gradient(g_i)와 Hessian(h_i)의 연산을 지원합니다.
    """
    def __init__(self):
        # 1. TenSEAL Context 초기화 (CKKS Scheme - 실수 연산용)
        # poly_modulus_degree: 8192 (데이터 안전성 보장 레벨)
        # coeff_mod_bit_sizes: 암호문 레벨 곱셈/덧셈 뎁스
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        
    def encrypt(self, vector: list[float]) -> ts.CKKSVector:
        """평문 리스트(Gradient, Hessian 등)를 동형암호화합니다."""
        return ts.ckks_vector(self.context, vector)

    def decrypt(self, enc_vector: ts.CKKSVector) -> list[float]:
        """[Active Party 전용] 암호화된 벡터(히스토그램 등)를 평문으로 복호화합니다."""
        return enc_vector.decrypt()
    
    def add(self, enc_vec1: ts.CKKSVector, enc_vec2: ts.CKKSVector) -> ts.CKKSVector:
        """암호 상태에서의 덧셈 (Paillier Additive Homomorphism 수식 적용)"""
        return enc_vec1 + enc_vec2
