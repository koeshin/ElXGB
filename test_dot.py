import tenseal as ts
import numpy as np

def test():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()

    print("Testing length 4000 (single ciphertext)...")
    data1 = np.random.rand(4000).tolist()
    enc1 = ts.ckks_vector(context, data1)
    mask1 = np.random.randint(0, 2, 4000).tolist()
    try:
        res1 = enc1.dot(mask1)
        print("4000 int mask: Success")
    except Exception as e:
        print("4000 int mask Failed:", e)

    print("\nTesting length 10000 (multi ciphertext)...")
    data2 = np.random.rand(10000).tolist()
    enc2 = ts.ckks_vector(context, data2)
    
    mask2_int = np.random.randint(0, 2, 10000).tolist()
    try:
        res2 = enc2.dot(mask2_int)
        print("10000 int mask: Success")
    except Exception as e:
        print("10000 int mask Failed:", e)
        
    mask2_float = np.random.randint(0, 2, 10000).astype(float).tolist()
    try:
        res2 = enc2.dot(mask2_float)
        print("10000 float mask: Success")
    except Exception as e:
        print("10000 float mask Failed:", e)

if __name__ == "__main__":
    test()
