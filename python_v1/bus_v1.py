import numpy as np
from typing import List, Tuple, Dict, Any
from ctypes import *
from puf_gen import key_gen
from Crypto.PublicKey import ECC
from Crypto.Hash import SHA256
from Crypto.Cipher import AES

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
    arr = np.array(flat_weights[start:end], dtype=np.float32).reshape(shape)
    return arr


TX_SIZE = 256

def ecc_key_construct():
    puf_out = key_gen(0x9E4b, 12, 0x5C17A6D3)
    puf_key = bytearray(puf_out)
    d = int.from_bytes(puf_key, byteorder='big')
    private_key = ECC.construct(curve='P-256', d=d)
    public_key = private_key.export_key(format='PEM')
    return private_key, public_key


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

def encrypt_aes(chunk: List[float], puf_key: bytes) -> Tuple[List[float], bytes, bytes, int]:
    plaintext_bytes = floats_to_bytes(chunk)
    original_len = len(plaintext_bytes)
    cipher = AES.new(puf_key, AES.MODE_GCM)
    ciphertext_bytes, tag = cipher.encrypt_and_digest(plaintext_bytes)
    encrypted_chunk = bytes_to_floats(ciphertext_bytes)
    return encrypted_chunk, tag, cipher.nonce, original_len

def decrypt_aes(encrypted_chunk: List[float], puf_key: bytes, tag: bytes, nonce: bytes, original_len: int = None) -> Tuple[List[float], bool]:
    try:
        ciphertext_bytes = floats_to_bytes(encrypted_chunk)
        # Trim to original length if padding was added during bytes_to_floats conversion
        # This is critical - we must decrypt exactly the original ciphertext length
        if original_len is not None and len(ciphertext_bytes) > original_len:
            ciphertext_bytes = ciphertext_bytes[:original_len]
        cipher = AES.new(puf_key, AES.MODE_GCM, nonce=nonce)
        plaintext_bytes = cipher.decrypt_and_verify(ciphertext_bytes, tag)
        # Trim plaintext to original length if needed (should match, but be safe)
        if original_len is not None and len(plaintext_bytes) > original_len:
            plaintext_bytes = plaintext_bytes[:original_len]
        decrypted_chunk = bytes_to_floats(plaintext_bytes)
        return decrypted_chunk, True
    except ValueError:
        return encrypted_chunk, False

def fault_injection(enc_weight: float) -> float:
    flipped_weight: float = 0.0
    rng = np.random.default_rng()
    rint = rng.integers(low=0, high=63, size=1)

    bits = cast(pointer(c_double(enc_weight)), POINTER(c_uint64)).contents.value
    bits ^= int(1 << rint[0])
    flipped_weight = cast(pointer(c_uint64(bits)), POINTER(c_double)).contents.value
    return flipped_weight

def transmit_over_bus(data: List[float], num_flips=0, puf_key: bytes = None) -> Tuple[List[float], List[List[float]]]:
    if puf_key is None:
        puf_key = key_gen(0x9E4b, 0, 0x5C17A6D3)
    
    transmitted: List[float] = []
    bus_log: List[List[float]] = []
    flip_ctr = num_flips if (num_flips < TX_SIZE) else (TX_SIZE-1)
    
    for i in range(0, len(data), TX_SIZE):
        chunk = data[i:i + TX_SIZE]
        enc_chunk, tag, nonce, original_len = encrypt_aes(chunk, puf_key)
        if(flip_ctr != 0):
            enc_chunk[0] = fault_injection(enc_chunk[0])
            flip_ctr -= 1
        bus_log.append(enc_chunk)
        dec_chunk, verified = decrypt_aes(enc_chunk, puf_key, tag, nonce, original_len)
        if not verified:
            print("data tampered")
        transmitted.extend(dec_chunk)
    return transmitted, bus_log


if __name__ == "__main__":
    # fake MNIST weights
    weights, layout = create_small_mnist_weights(seed=123)
    print(f"total values generated: {len(weights)}")

    # transmission (unpack both returned values)
    # set num_flips=0 for testing
    received, bus_log = transmit_over_bus(weights, num_flips=0)
    print(f"total values received: {len(received)}")
    print(f"chunks captured on bus: {len(bus_log)}")

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
