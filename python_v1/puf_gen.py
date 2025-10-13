# PUF generation - made to be similar to RTL design
# takes 32 bit challenge and outputs 128 bit response to be hashed
# accesses same wordline per "bank" - will improve performance in RTL
# in RTL, would open all bitlines of a bank, open given wordline, compare adjacent bitline pairs
# using SET race - shown in paper to be more reliable

# would be created in addition to array, not underneath it


import numpy as np
import hashlib

#array setup
N_ROWS = 4
N_COLS = 256
BANK_COLS = 16
N_BANKS = N_COLS // BANK_COLS  # 16 banks
PAIRS_PER_BANK = BANK_COLS // 2  # 8 pairs/bank
TOTAL_PAIRS = N_BANKS * PAIRS_PER_BANK  # 128 bits

#randomize t_set
def program_traces(seed=1234, mean_ns=100.0, sigma_ns=15.0):
    rng = np.random.default_rng(seed)
    t_set = rng.normal(loc=mean_ns, scale=sigma_ns, size=(N_ROWS, N_COLS))
    return np.clip(t_set, 60.0, 160.0)

#select row to use per bank
def decode_bank_row(chal_32: int, bank_id: int) -> int:
    shift = 2 * bank_id
    return (chal_32 >> shift) & 0b11

#race function
def race_bit(tset: np.ndarray, row: int, left_col: int) -> int:
    c = left_col
    tL = tset[row, c]
    tR = tset[row, c + 1]
    return int(tL < tR)

#use race bits to create bank structure
def bank_response_bits(tset: np.ndarray, row: int, base_col: int) -> list[int]:
    bits = []
    for i in range(0, BANK_COLS, 2):
        bits.append(race_bit(tset, row, base_col + i))
    return bits

#use 16 banks to generate 128 bit response
def response_bits(tset: np.ndarray, chal_32: int) -> list[int]:
    bits = []
    for b in range(N_BANKS):
        row_sel = decode_bank_row(chal_32, b)
        base_col = b * BANK_COLS
        bits += bank_response_bits(tset, row_sel, base_col)
    assert len(bits) == TOTAL_PAIRS
    return bits

#apply hash - step required in PUF -> key transition
def key_from_bits(bits: list[int]) -> bytes:
    assert len(bits) == 128
    byts = bytearray(16)
    for i, b in enumerate(bits):
        if b:
            byts[i // 8] |= 1 << (7 - (i % 8))
    digest = hashlib.sha256(bytes(byts)).digest()
    return digest[:16]


if __name__ == "__main__":
    tset = program_traces(seed=0x9E4B)
    chal = 0x5C17A6D3

    bits = response_bits(tset, chal)
    key = key_from_bits(bits)

    print(f"32-bit challenge = 0x{chal:08X}")
    print(f"raw response vector (128): {''.join(map(str,bits))}")
    print(f"128-bit key: {key.hex()}")
    print(f"bit balance: ones={sum(bits)}, zeros={128 - sum(bits)}")
