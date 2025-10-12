import numpy as np
import pandas as pd
import hashlib
import random

rows, cols = 4, 256
np.random.seed(44)

data = np.random.randint(0, 2, size=(rows, cols))

t_set   = np.clip(np.random.normal(loc=100, scale=20, size=(rows, cols)), 50, 150)
t_reset = np.clip(np.random.normal(loc=120, scale=25, size=(rows, cols)), 60, 180)


cells = pd.DataFrame({
    'row': np.repeat(np.arange(rows), cols),
    'col': np.tile(np.arange(cols), rows),
    'data': data.flatten(),
    't_set': t_set.flatten(),
    't_reset': t_reset.flatten()
})


def prng_expand(challenge_seed, n_pairs, n_rows, n_cols):
    rng = random.Random(challenge_seed)
    pairs = []
    for _ in range(n_pairs):
        #pick two cells
        a_row, a_col = rng.randrange(n_rows), rng.randrange(n_cols)
        b_row, b_col = rng.randrange(n_rows), rng.randrange(n_cols)
        while a_row == b_row and a_col == b_col:
            b_row, b_col = rng.randrange(n_rows), rng.randrange(n_cols)
        #pick set or reset race
        mode = rng.randint(0, 1)
        pairs.append(((a_row, a_col), (b_row, b_col), mode))
    return pairs


def race_bit(cells_df, addrA, addrB, mode):
    idxA = addrA[0] * cols + addrA[1]
    idxB = addrB[0] * cols + addrB[1]
    if mode == 0:  # SET race
        tA, tB = cells_df.loc[idxA, 't_set'], cells_df.loc[idxB, 't_set']
    else:          # RESET race
        tA, tB = cells_df.loc[idxA, 't_reset'], cells_df.loc[idxB, 't_reset']
    bit = 1 if tA < tB else 0
    margin = abs(tA - tB)
    return bit, margin


def derive_key(bitstring, key_bits=128):
    byte_data = int(''.join(str(b) for b in bitstring), 2).to_bytes(len(bitstring)//8, 'big', signed=False)
    digest = hashlib.sha256(byte_data).digest()
    return digest[: key_bits // 8]


NUM_CHALS = 32
PAIRS_PER_CHAL = 16
GUARD_BAND = 10.0

raw_bits = []
for chal_idx in range(NUM_CHALS):
    challenge_seed = np.random.randint(0, 2**32)
    pairs = prng_expand(challenge_seed, PAIRS_PER_CHAL, rows, cols)
    for (a, b, mode) in pairs:
        bit, margin = race_bit(cells, a, b, mode)
        if margin >= GUARD_BAND:
            raw_bits.append(bit)
# ensure length multiple of 8 for hashing
if len(raw_bits) % 8 != 0:
    raw_bits = raw_bits[: len(raw_bits) - (len(raw_bits) % 8)]

print(f"Generated {len(raw_bits)} reliable race bits.")


key = derive_key(raw_bits, key_bits=128)
print("PUF-derived key (128-bit):", key.hex())


import matplotlib.pyplot as plt

plt.hist(cells['t_set'], bins=30, alpha=0.6, label='SET times')
plt.hist(cells['t_reset'], bins=30, alpha=0.6, label='RESET times')
plt.xlabel("Time (ns)")
plt.ylabel("Count")
plt.title("Distribution of SET/RESET Times")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
