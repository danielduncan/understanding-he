# A toy SHE implementation guided by https://bit-ml.github.io/blog/post/homomorphic-encryption-toy-implementation-in-python/

import numpy as np
from numpy.polynomial import polynomial as poly

def poly_mul_mod(x, y, poly_mod):
    return poly.polydiv(poly.polymul(x, y), poly_mod)[1]

def poly_add_mod(x, y, poly_mod):
    return poly.polydiv(poly.polyadd(x, y), poly_mod)[1]

def polymul(x, y, mod, poly_mod):
    return np.int64(np.round(poly.polydiv(poly.polymul(x, y) % mod, poly_mod)[1] % mod))

def polyadd(x, y, mod, poly_mod):
    return np.int64(np.round(poly.polydiv(poly.polyadd(x, y) % mod, poly_mod)[1] % mod))

def binary_poly(size):
    return np.random.randint(0, 2, size, dtype=np.int64) # array of coefficients in [0, 1]

def uniform_poly(size, mod):
    return np.random.randint(0, mod, size, dtype=np.int64) # array of coefficients in Z mod

def normal_poly(size, mean, std):
    return np.int64(np.random.normal(mean, std, size)) # array of normally distributed coefficients

def keygen(size, mod, poly_mod, std):
    s = binary_poly(size)
    a = uniform_poly(size, mod)
    e = normal_poly(size, 0, std)
    b = polyadd(polymul(-a, s, mod, poly_mod), -e, mod, poly_mod) # [-(a*s + e)]_q = [-a*s + -e]_q
    return (b, a), s # pk = ([-(a*s + e)]_q, a), sk = s

def encrypt(pk, m, size, cipher_mod, plain_mod, poly_mod, std):
    m = np.array(m + [0] * (size - len(m)), dtype=np.int64) % plain_mod # pad m with zeros and encode in R_plain_mod
    delta = cipher_mod // plain_mod
    scaled_m = m * delta
    e1 = normal_poly(size, 0, std)
    e2 = normal_poly(size, 0, std)
    u = uniform_poly(size, 2)
    ct0 = polyadd(polyadd(polymul(pk[0], u, cipher_mod, poly_mod), e1, cipher_mod, poly_mod), scaled_m, cipher_mod, poly_mod)
    ct1 = polyadd(polymul(pk[1], u, cipher_mod, poly_mod), e2, cipher_mod, poly_mod)
    return (ct0, ct1)

def padded(poly, size):
    if len([i for i in poly]) < size:
        return np.append(poly, [0] * (size - len(poly)))
    else:
        return poly

def decrypt(sk, ct, cipher_mod, plain_mod, poly_mod, size):
    cts = polyadd(ct[0], polymul(ct[1], sk, cipher_mod, poly_mod), cipher_mod, poly_mod) # scaled plaintext + noise i.e. delta * m + v
    decrypt_poly = np.round((plain_mod * cts) / cipher_mod) % plain_mod
    return np.int64(padded(decrypt_poly, size))

def add(ct1, ct2, cipher_mod, poly_mod):
    add_ct0 = polyadd(ct1[0], ct2[0], cipher_mod, poly_mod)
    add_ct1 = polyadd(ct1[1], ct2[1], cipher_mod, poly_mod)
    return (add_ct0, add_ct1)

def mul(ct1, ct2, poly_mod, cipher_mod, plain_mod):
    c0x = poly_mul_mod(ct1[0], ct2[0], poly_mod)
    c1x = poly_add_mod(poly_mul_mod(ct1[0], ct2[1], poly_mod), poly_mul_mod(ct1[1], ct2[0], poly_mod), poly_mod)
    c2x = poly_mul_mod(ct1[1], ct2[1], poly_mod)
    c_0 = np.int64(np.round(c0x * (plain_mod / cipher_mod))) % cipher_mod
    c_1 = np.int64(np.round(c1x * (plain_mod / cipher_mod))) % cipher_mod
    c_2 = np.int64(np.round(c2x * (plain_mod / cipher_mod))) % cipher_mod
    return c_0, c_1, c_2

def eval_keygen(sk, size, mod, poly_mod, extra_mod, std):
    new_mod = mod * extra_mod
    a = uniform_poly(size, new_mod)
    e = normal_poly(size, 0, std)
    secret = extra_mod * poly.polymul(sk, sk) # c2 * s^2
    
    b = np.int64(poly_add_mod(poly_mul_mod(-a, sk, poly_mod), poly_add_mod(-e, secret, poly_mod), poly_mod)) % new_mod # [-(a * s  + e) + p * s^2]_(p * q)
    return b, a # [-(a * s  + e) + p * s^2]_(p * q), a

def mul_cipher(ct1, ct2, cipher_mod, plain_mod, switch_mod, poly_mod, rlk0, rlk1):
    c_0, c_1, c_2 = mul(ct1, ct2, poly_mod, cipher_mod, plain_mod)

    c_20 = np.int64(np.round(poly_mul_mod(c_2, rlk0, poly_mod) / switch_mod)) % cipher_mod
    c_21 = np.int64(np.round(poly_mul_mod(c_2, rlk1, poly_mod) / switch_mod)) % cipher_mod

    new_c0 = np.int64(poly_add_mod(c_0, c_20, poly_mod)) % cipher_mod # c0 + c20
    new_c1 = np.int64(poly_add_mod(c_1, c_21, poly_mod)) % cipher_mod # c1 + c21

    return (new_c0, new_c1) # cmul = (c0 + c20, c1 + c21)