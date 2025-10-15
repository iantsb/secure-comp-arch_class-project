from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import time
import hashlib
from puf_gen import key_gen

def encrypt_rsa(public_key, plaintext):
    cipher = PKCS1_OAEP.new(public_key)
    ciphertext = cipher.encrypt(plaintext)
    print("encryption done: ciphertext:", ciphertext.hex()[:64])
    return ciphertext

def decrypt_rsa(private_key, ciphertext):
    cipher = PKCS1_OAEP.new(private_key)
    plaintext = cipher.decrypt(ciphertext)
    print("decrypt done: ", plaintext)
    return plaintext

# Generate RSA keypair from PUF
puf_output = key_gen(0x9E4b, 0, 0x5C17A6D3)
print("PUF output:", puf_output.hex())

# Use PUF output to seed deterministic random function for RSA generation
class DeterministicRandom:
    def __init__(self, seed):
        self.seed = seed
        self.counter = 0
    
    def __call__(self, n):
        result = b''
        while len(result) < n:
            data = self.seed + self.counter.to_bytes(8, 'big')
            result += hashlib.sha256(data).digest()
            self.counter += 1
        return result[:n]

randfunc = DeterministicRandom(puf_output)
rsa_key = RSA.generate(2048, randfunc=randfunc)
public_key = rsa_key.publickey()

message = b"test"
print("RSA public key (n):", hex(public_key.n)[:50])

encrypt_start = time.time()
ciphertext = encrypt_rsa(public_key, message)
encrypt_end = time.time()

decrypt_start = time.time()
decrypt_rsa(rsa_key, ciphertext)
decrypt_end = time.time()

print("encryption took: ", encrypt_end-encrypt_start)
print("decryption took: ", decrypt_end-decrypt_start)

