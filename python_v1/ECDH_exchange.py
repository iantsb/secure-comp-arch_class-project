from Crypto.PublicKey import ECC
import time
import hashlib
from puf_gen import key_gen

def generate_ecc_key(puf_seed, puf_challenge):
    puf_output = key_gen(puf_seed, 0, puf_challenge)
    # Use PUF output to deterministically generate ECC private key
    seed_hash = hashlib.sha256(puf_output).digest()
    d = int.from_bytes(seed_hash, byteorder='big')
    private_key = ECC.construct(curve='P-256', d=d)
    return private_key, puf_output

def ecdh_derive_shared(my_private_key, their_public_key):
    # ECDH: multiply my private scalar with their public point
    shared_point = my_private_key.d * their_public_key.pointQ
    # Use x-coordinate as shared secret
    shared_secret = shared_point.x.to_bytes(32, byteorder='big')
    # Hash to get final key
    return hashlib.sha256(shared_secret).digest()[:16]

# Party A generates keypair from PUF
print("Party A generating keypair from PUF...")
keygen_a_start = time.time()
private_key_a, puf_a = generate_ecc_key(0x9E4b, 0x5C17A6D3)
public_key_a = private_key_a.public_key()
keygen_a_end = time.time()
print("Party A PUF output:", puf_a.hex())
print("Party A public key:", public_key_a.export_key(format='DER').hex()[:32])

# Party B generates keypair from PUF
print("Party B generating keypair from PUF...")
keygen_b_start = time.time()
private_key_b, puf_b = generate_ecc_key(0xAAAA, 0x12345678)
public_key_b = private_key_b.public_key()
keygen_b_end = time.time()
print("Party B PUF output:", puf_b.hex())
print("Party B public key:", public_key_b.export_key(format='DER').hex()[:32])

# Party A derives shared secret
derive_a_start = time.time()
shared_key_a = ecdh_derive_shared(private_key_a, public_key_b)
derive_a_end = time.time()
print("Party A derived key:", shared_key_a.hex())

# Party B derives shared secret
derive_b_start = time.time()
shared_key_b = ecdh_derive_shared(private_key_b, public_key_a)
derive_b_end = time.time()
print("Party B derived key:", shared_key_b.hex())

print("Keys match:", shared_key_a == shared_key_b)

print("Party A keygen took: ", keygen_a_end-keygen_a_start)
print("Party B keygen took: ", keygen_b_end-keygen_b_start)
print("Party A ECDH derive took: ", derive_a_end-derive_a_start)
print("Party B ECDH derive took: ", derive_b_end-derive_b_start)
