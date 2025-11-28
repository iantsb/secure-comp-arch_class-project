import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from ctypes import *
from puf_gen import key_gen
from Crypto.PublicKey import ECC, RSA
from Crypto.Hash import SHA256
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from functools import partial
import hashlib

def create_small_mnist_weights(seed: int = 0) -> Tuple[List[float], Dict[str, Tuple[tuple, int, int]]]:
    rng = np.random.default_rng(seed)

    shapes = {
        "W1": (64, 784),
        "b1": (64,),
        "W2": (10, 64),
        "b2": (10,),
    }

    layout: Dict[str, Tuple[tuple, int, int]] = {}
    flat_weights: List[float] = []
    current = 0

    for name, shape in shapes.items():
        size = int(np.prod(shape))
        data = rng.normal(0.0, 0.1, size).astype(np.float32)
        layout[name] = (shape, current, current + size)
        flat_weights.extend(data.tolist())
        current += size

    return flat_weights, layout


def reconstruct_layer(flat_weights: List[float], layout_entry: Tuple[tuple, int, int]) -> np.ndarray:
    shape, start, end = layout_entry
    arr = np.array(flat_weights[start:end], dtype=np.float64).reshape(shape)
    return arr


TX_SIZE = 256

def ecc_key_construct(puf_seed=0x9E4b, puf_challenge=0x5C17A6D3):
    puf_output = key_gen(puf_seed, 0, puf_challenge)
    
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
    private_key = ECC.generate(curve='P-256', randfunc=randfunc)
    public_key = private_key.public_key()
    return private_key, public_key

def rsa_key_construct(puf_seed=0x9E4b, puf_challenge=0x5C17A6D3):
    puf_output = key_gen(puf_seed, 0, puf_challenge)
    
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
    return rsa_key, rsa_key.publickey()

def ecdh_derive_shared(my_private_key, their_public_key):
    shared_point = my_private_key.d * their_public_key.pointQ
    shared_secret = shared_point.x.to_bytes(32, byteorder='big')
    return hashlib.sha256(shared_secret).digest()[:16]


# def encrypt_placeholder(chunk: List[float]) -> Tuple[List[float], str]:
#     # encryption scheme (no-op for now)
#     h = SHA256.new()
#     to_hash = bytearray()
#     for i in range(len(chunk)):
#         bits = cast(pointer(c_double(chunk[i])), POINTER(c_uint64)).contents.value
#         to_hash.extend(bits.to_bytes(8, byteorder='big'))
#     h.update(to_hash)
#     return chunk, h.hexdigest()


# def decrypt_placeholder(chunk: List[float]) -> Tuple[List[float], str]:
#     # decryption scheme (no-op for now)
#     h = SHA256.new()
#     to_hash = bytearray()
#     for i in range(len(chunk)):
#         bits = cast(pointer(c_double(chunk[i])), POINTER(c_uint64)).contents.value
#         to_hash.extend(bits.to_bytes(8, byteorder='big'))
#     h.update(to_hash)
#     return chunk, h.hexdigest()

def floats_to_bytes(chunk: List[float]) -> bytes:
    byte_data = bytearray()
    for f in chunk:
        bits = cast(pointer(c_double(f)), POINTER(c_uint64)).contents.value
        byte_data.extend(bits.to_bytes(8, byteorder='big'))
    return bytes(byte_data)

def bytes_to_floats(byte_data: bytes) -> List[float]:
    chunk = []
    # Ensure byte_data length is a multiple of 8 (should always be for float conversion)
    # If not, pad with zeros (though this shouldn't happen with AES-GCM)
    byte_len = len(byte_data)
    if byte_len % 8 != 0:
        padding_needed = 8 - (byte_len % 8)
        byte_data = byte_data + b'\x00' * padding_needed
    
    for i in range(0, len(byte_data), 8):
        bits = int.from_bytes(byte_data[i:i+8], byteorder='big')
        f = cast(pointer(c_uint64(bits)), POINTER(c_double)).contents.value
        # Preserve exact bit pattern, even if it's NaN/Inf (needed for encryption)
        chunk.append(f)
    return chunk

def encrypt_aes(chunk: List[float], puf_key: bytes) -> Tuple[List[float], bytes, bytes, bytes, int]:
    plaintext_bytes = floats_to_bytes(chunk)
    original_len = len(plaintext_bytes)
    cipher = AES.new(puf_key, AES.MODE_GCM)
    ciphertext_bytes, tag = cipher.encrypt_and_digest(plaintext_bytes)
    encrypted_chunk = bytes_to_floats(ciphertext_bytes)
    return encrypted_chunk, tag, cipher.nonce, b'', original_len  # metadata empty for AES

def decrypt_aes(encrypted_chunk: List[float], puf_key: bytes, tag: bytes, nonce: bytes, metadata: bytes, original_len: int = None) -> Tuple[List[float], bool]:
    try:
        ciphertext_bytes = floats_to_bytes(encrypted_chunk)
        if original_len is not None and len(ciphertext_bytes) > original_len:
            ciphertext_bytes = ciphertext_bytes[:original_len]
        cipher = AES.new(puf_key, AES.MODE_GCM, nonce=nonce)
        plaintext_bytes = cipher.decrypt_and_verify(ciphertext_bytes, tag)
        if original_len is not None and len(plaintext_bytes) > original_len:
            plaintext_bytes = plaintext_bytes[:original_len]
        decrypted_chunk = bytes_to_floats(plaintext_bytes)
        return decrypted_chunk, True
    except ValueError:
        return encrypted_chunk, False
def encrypt_ecc(chunk: List[float], recipient_public_key: ECC.EccKey) -> Tuple[List[float], bytes, bytes, bytes, int]:
    plaintext_bytes = floats_to_bytes(chunk)
    original_len = len(plaintext_bytes)
    
    # Generate ephemeral keypair
    ephemeral_private = ECC.generate(curve='P-256')
    ephemeral_public = ephemeral_private.public_key()
    
    # Derive shared secret using ECDH
    shared_secret = ecdh_derive_shared(ephemeral_private, recipient_public_key)
    
    # Encrypt with AES-GCM using shared secret
    cipher = AES.new(shared_secret, AES.MODE_GCM)
    ciphertext_bytes, tag = cipher.encrypt_and_digest(plaintext_bytes)
    encrypted_chunk = bytes_to_floats(ciphertext_bytes)
    
    # Store ephemeral public key as metadata
    ephemeral_pub_bytes = ephemeral_public.export_key(format='DER')
    
    return encrypted_chunk, tag, cipher.nonce, ephemeral_pub_bytes, original_len

def decrypt_ecc(encrypted_chunk: List[float], private_key: ECC.EccKey, tag: bytes, nonce: bytes, metadata: bytes, original_len: int = None) -> Tuple[List[float], bool]:
    try:
        # Reconstruct ephemeral public key from metadata
        ephemeral_public = ECC.import_key(metadata)
        
        # Derive shared secret using ECDH
        shared_secret = ecdh_derive_shared(private_key, ephemeral_public)
        
        # Decrypt with AES-GCM
        ciphertext_bytes = floats_to_bytes(encrypted_chunk)
        if original_len is not None and len(ciphertext_bytes) > original_len:
            ciphertext_bytes = ciphertext_bytes[:original_len]
        cipher = AES.new(shared_secret, AES.MODE_GCM, nonce=nonce)
        plaintext_bytes = cipher.decrypt_and_verify(ciphertext_bytes, tag)
        if original_len is not None and len(plaintext_bytes) > original_len:
            plaintext_bytes = plaintext_bytes[:original_len]
        decrypted_chunk = bytes_to_floats(plaintext_bytes)
        return decrypted_chunk, True
    except (ValueError, Exception):
        return encrypted_chunk, False

def encrypt_rsa(chunk: List[float], recipient_public_key: RSA.RsaKey) -> Tuple[List[float], bytes, bytes, bytes, int]:
    plaintext_bytes = floats_to_bytes(chunk)
    original_len = len(plaintext_bytes)
    
    # RSA-2048 with PKCS1_OAEP can encrypt max 214 bytes at a time (256 - 42 for padding)
    max_block_size = recipient_public_key.size_in_bytes() - 42  # OAEP padding overhead
    
    # Encrypt each block with RSA
    cipher_rsa = PKCS1_OAEP.new(recipient_public_key)
    encrypted_blocks = []
    
    for i in range(0, len(plaintext_bytes), max_block_size):
        block = plaintext_bytes[i:i + max_block_size]
        encrypted_block = cipher_rsa.encrypt(block)
        encrypted_blocks.append(encrypted_block)
    
    # Concatenate all encrypted blocks
    ciphertext_bytes = b''.join(encrypted_blocks)
    
    # Convert encrypted bytes back to floats
    encrypted_chunk = bytes_to_floats(ciphertext_bytes)
    
    # Store number of blocks in metadata for decryption
    num_blocks = len(encrypted_blocks)
    metadata = num_blocks.to_bytes(4, byteorder='big')
    
    # Return empty tag/nonce since RSA doesn't use them (using empty bytes for compatibility)
    return encrypted_chunk, b'', b'', metadata, original_len

def decrypt_rsa(encrypted_chunk: List[float], private_key: RSA.RsaKey, tag: bytes, nonce: bytes, metadata: bytes, original_len: int = None) -> Tuple[List[float], bool]:
    try:
        # Extract number of blocks from metadata (must be 4 bytes)
        if len(metadata) < 4:
            # Return placeholder data with correct length if original_len is known
            if original_len is not None:
                num_floats = original_len // 8
                return [0.0] * num_floats, False
            return encrypted_chunk, False
        num_blocks = int.from_bytes(metadata[:4], byteorder='big')
        
        # Convert encrypted chunk back to bytes
        ciphertext_bytes = floats_to_bytes(encrypted_chunk)
        
        # RSA-2048 encrypted blocks are 256 bytes each
        block_size = private_key.size_in_bytes()
        
        # Decrypt each block with RSA
        cipher_rsa = PKCS1_OAEP.new(private_key)
        decrypted_blocks = []
        
        for i in range(num_blocks):
            block_start = i * block_size
            block_end = block_start + block_size
            encrypted_block = ciphertext_bytes[block_start:block_end]
            
            decrypted_block = cipher_rsa.decrypt(encrypted_block)
            decrypted_blocks.append(decrypted_block)
        
        # Concatenate all decrypted blocks
        plaintext_bytes = b''.join(decrypted_blocks)
        
        # Trim to original length if specified
        if original_len is not None and len(plaintext_bytes) > original_len:
            plaintext_bytes = plaintext_bytes[:original_len]
        
        # Convert back to floats
        decrypted_chunk = bytes_to_floats(plaintext_bytes)
        return decrypted_chunk, True
    except (ValueError, Exception):
        print('data tampered, rsa decryption failed, returning garbage data')
        if original_len is not None:
            num_floats = original_len // 8
            return [0.0] * num_floats, False
        # Fallback: return empty list if we can't determine original length
        return [], False

def fault_injection(enc_weight: float) -> float:
    flipped_weight: float = 0.0
    rng = np.random.default_rng()
    rint = rng.integers(low=0, high=63, size=1)

    bits = cast(pointer(c_double(enc_weight)), POINTER(c_uint64)).contents.value
    bits ^= int(1 << rint[0])
    flipped_weight = cast(pointer(c_uint64(bits)), POINTER(c_double)).contents.value
    return flipped_weight

def transmit_over_bus(data: List[float], encryption_mode, num_flips=0,puf_key: Optional[bytes] = None) -> Tuple[List[float], List[List[float]]]:
    puf_seed = 0x9E4b
    puf_challenge = 0x5C17A6D3
    # Initialize encryption based on mode
    if encryption_mode == 'aes':
        if puf_key is None:
            puf_key = key_gen(puf_seed, 0, puf_challenge)
        encrypt_func = partial(encrypt_aes, puf_key=puf_key)
        decrypt_func = partial(decrypt_aes, puf_key=puf_key)
    elif encryption_mode == 'ecc':
        private_key, public_key = ecc_key_construct(puf_seed, puf_challenge)
        encrypt_func = partial(encrypt_ecc, recipient_public_key=public_key)
        decrypt_func = partial(decrypt_ecc, private_key=private_key)
    elif encryption_mode == 'rsa':
        private_key, public_key = rsa_key_construct(puf_seed, puf_challenge)
        encrypt_func = partial(encrypt_rsa, recipient_public_key=public_key)
        decrypt_func = partial(decrypt_rsa, private_key=private_key)
    else:
        raise ValueError(f"Unknown encryption_mode: {encryption_mode}. Must be 'aes', 'ecc', or 'rsa'")
    
    transmitted: List[float] = []
    bus_log: List[List[float]] = []
    flip_ctr = num_flips if (num_flips < TX_SIZE) else (TX_SIZE-1)
    
    for i in range(0, len(data), TX_SIZE):
        chunk = data[i:i + TX_SIZE]
        enc_chunk, tag, nonce, metadata, original_len = encrypt_func(chunk)
        if(flip_ctr != 0):
            enc_chunk[0] = fault_injection(enc_chunk[0])
            flip_ctr -= 1
        bus_log.append(enc_chunk)
        dec_chunk, verified = decrypt_func(encrypted_chunk=enc_chunk, tag=tag, nonce=nonce, metadata=metadata, original_len=original_len)
        if not verified:
            print("data tampered")
        transmitted.extend(dec_chunk)
    return transmitted, bus_log


if __name__ == "__main__":
    # fake MNIST weights
    weights, layout = create_small_mnist_weights(seed=123)
    print(f"total values generated: {len(weights)}")

    encryption_mode = 'ecc'
    
    # transmission (unpack both returned values)
    received, bus_log = transmit_over_bus(weights, num_flips=0, encryption_mode=encryption_mode)
    print(f"total values received: {len(received)}")
    print(f"chunks captured on bus: {len(bus_log)}")
    print(f"encryption mode used: {encryption_mode}")

    # check integrity
    if np.allclose(received, weights):
        print("data matches")
    else:
        print("data tampered")

    # Reconstruct layers from received floats
    W1 = reconstruct_layer(received, layout["W1"])
    b1 = reconstruct_layer(received, layout["b1"])
    W2 = reconstruct_layer(received, layout["W2"])
    b2 = reconstruct_layer(received, layout["b2"])

    print("Layer shapes after reconstruction:")
    print("W1:", W1.shape, "b1:", b1.shape)
    print("W2:", W2.shape, "b2:", b2.shape)
