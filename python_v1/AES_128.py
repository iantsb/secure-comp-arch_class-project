from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import time
from puf_gen import key_gen

def encrypt_msg(key, plaintxt, header):
    cipher = AES.new(key, AES.MODE_GCM)
    cipher.update(header)
    ciphertext, tag = cipher.encrypt_and_digest(plaintxt)
    print("encryption done: ciphertext:", ciphertext)
    return ciphertext, tag, cipher.nonce 

def decrypt_msg(key, ciphertext, tag, nonce, header):
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(header)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    print("decrypt done: ", plaintext)
    return plaintext

aes_128_key = key_gen(0x9E4b, 0, 0x5C17A6D3)
# function used to generate relevant key
header = b"header"
message = b"puf encryption is so much fun!"
print(aes_128_key.hex())
encrypt_start = time.time()
ciphertext, tag, nonce = encrypt_msg(aes_128_key, message, header)
encrypt_end = time.time()

decrypt_start = time.time()
decrypt_msg(aes_128_key, ciphertext, tag, nonce, header)
decrypt_end = time.time()

print("encryption took: ", encrypt_end-encrypt_start)
print("decryption took: ", decrypt_end-decrypt_start)