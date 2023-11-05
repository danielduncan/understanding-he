import she
import numpy as np
import time

def bintodec(bin):
    return sum(val*(2**idx) for idx, val in enumerate(bin))

n = 2 ** 2 # polynomial modulus degree (i.e. 2 which is a power of 2)
q = 2 ** 14 # ciphertext modulus
t = 2 # plaintext modulus
p = q ** 3 # switching modulus
poly_mod = np.array([1] + [0] * (n - 1) + [1])
std_err_enc = 1 # std dev of error in encryption
std_err_eval = 1 # std dev of error in evaluation keygen

pk, sk = she.keygen(n, q, poly_mod, std_err_enc)

rlk0, rlk1, = she.eval_keygen(sk, n, q, poly_mod, p, std_err_eval)

print("Plaintext -> Ciphertext")
pt1, pt2 = [1, 0, 1, 0], [0, 1, 0, 0] # 5, 2
ct1 = she.encrypt(pk, pt1, n, q, t, poly_mod, std_err_enc)
ct2 = she.encrypt(pk, pt2, n, q, t, poly_mod, std_err_enc)
print("\t A: {} -> {}".format(pt1, ct1[0]))
print("\t B: {} -> {}".format(pt2, ct2[0]))

print("A + B = C")
ct3 = she.add(ct1, ct2, q, poly_mod)
print("\t Plaintext: {} + {} = {}".format(bintodec(pt1), bintodec(pt2), bintodec(pt1) + bintodec(pt2)))
print("\t Ciphertext: {} + {} = {}".format(ct1[0], ct2[0], ct3[0]))
print("\t Decrypted: {} + {} = {}".format(bintodec(she.decrypt(sk, ct1, q, t, poly_mod, 4)), bintodec(she.decrypt(sk, ct2, q, t, poly_mod, 4)), bintodec(she.decrypt(sk, ct3, q, t, poly_mod, 4))))

print("A * B = D")
ct4 = she.mul_cipher(ct1, ct2, q, t, p, poly_mod, rlk0, rlk1)
print("\t Plaintext: {} * {} = {}".format(bintodec(pt1), bintodec(pt2), bintodec(pt1) * bintodec(pt2)))
print("\t Ciphertext: {} * {} = {}".format(ct1[0], ct2[0], ct4[0]))
print("\t Decrypted: {} * {} = {}".format(bintodec(she.decrypt(sk, ct1, q, t, poly_mod, 4)), bintodec(she.decrypt(sk, ct2, q, t, poly_mod, 4)), bintodec(she.decrypt(sk, ct4, q, t, poly_mod, 4))))

pt3 = [1, 0, 0, 0] # 1
# Demo of inaccuracy due to noise growth
print("(1 * 1)^5")
ct5 = she.encrypt(pk, pt3, n, q, t, poly_mod, std_err_enc)
ct6 = she.mul_cipher(ct5, ct5, q, t, p, poly_mod, rlk0, rlk1)
for i in range(5):
    ct6 = she.mul_cipher(ct5, ct6, q, t, p, poly_mod, rlk0, rlk1)
    print("\t Multiplication {}: ({} * {})^10000 = {}".format(i, 1, 1, bintodec(she.decrypt(sk, ct6, q, t, poly_mod, 4))))

# Demo of how inefficient HE currently is
print("(1 * 1)^10000")
start = time.time()
pt4 = np.polymul(pt3, pt3)
for i in range(9999):
    pt4 = np.polymul(pt3, pt3)
end = time.time()
print("\t Plaintext ({}s): ({} * {})^10000 = {}".format(end - start, bintodec(pt3), bintodec(pt3), bintodec(pt4)))
ct7 = she.mul_cipher(ct5, ct5, q, t, p, poly_mod, rlk0, rlk1)
start = time.time()
for i in range(9999):
    ct7 = she.mul_cipher(ct5, ct7, q, t, p, poly_mod, rlk0, rlk1)
end = time.time()
print("\t Decrypted ({}s): ({} * {})^10000 = {}".format(end - start, bintodec(she.decrypt(sk, ct5, q, t, poly_mod, 4)), bintodec(she.decrypt(sk, ct5, q, t, poly_mod, 4)), bintodec(she.decrypt(sk, ct7, q, t, poly_mod, 4))))