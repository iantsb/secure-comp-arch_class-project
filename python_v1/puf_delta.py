import numpy as np
import matplotlib.pyplot as plt
from puf_gen import (
    key_gen,
    key_gen_round_based,
    DEFAULT_N_ROWS,
    DEFAULT_N_COLS,
    DEFAULT_BANK_COLS,
)

SEED = 0x16FB
CHALLENGE = 0x7D924EC2
NUM_RUNS = 1000
OFFSET_STEP = 0.1  # ns per iteration


N_ROWS = DEFAULT_N_ROWS
N_COLS = DEFAULT_N_COLS
BANK_COLS = DEFAULT_BANK_COLS


N_ROUNDS = 5
JITTER_SIGMA_NS = 0.5


def key_to_bits128(key_bytes: bytes) -> np.ndarray:
    kb = np.frombuffer(key_bytes, dtype=np.uint8)
    return np.unpackbits(kb)


def hamming_distance_bits(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a ^ b))



key_raw_single_ref = key_gen(
    seed=SEED,
    offset_ns=0.0,
    challenge=CHALLENGE,
    n_rows=N_ROWS,
    n_cols=N_COLS,
    bank_cols=BANK_COLS,
    use_hash=False,
)


key_raw_round_ref = key_gen_round_based(
    seed=SEED,
    offset_ns=0.0,
    challenge=CHALLENGE,
    n_rows=N_ROWS,
    n_cols=N_COLS,
    bank_cols=BANK_COLS,
    n_rounds=N_ROUNDS,
    jitter_sigma_ns=JITTER_SIGMA_NS,
    use_hash=False,
)


key_hash_single_ref = key_gen(
    seed=SEED,
    offset_ns=0.0,
    challenge=CHALLENGE,
    n_rows=N_ROWS,
    n_cols=N_COLS,
    bank_cols=BANK_COLS,
    use_hash=True,
)

ref_raw_single_bits = key_to_bits128(key_raw_single_ref)
ref_raw_round_bits = key_to_bits128(key_raw_round_ref)
ref_hash_single_bits = key_to_bits128(key_hash_single_ref)

raw_single_dists = []
raw_round_dists = []
hash_single_dists = []
offsets = np.arange(1, NUM_RUNS + 1) * OFFSET_STEP

for i in range(1, NUM_RUNS + 1):
    offset = i * OFFSET_STEP

    # Scheme 1: single-shot, raw (no hash)
    key_raw_single = key_gen(
        seed=SEED,
        offset_ns=offset,
        challenge=CHALLENGE,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        bank_cols=BANK_COLS,
        use_hash=False,
    )
    raw_single_bits = key_to_bits128(key_raw_single)
    raw_single_dists.append(
        hamming_distance_bits(ref_raw_single_bits, raw_single_bits)
    )

    # Scheme 2: round-based, raw (no hash)
    key_raw_round = key_gen_round_based(
        seed=SEED,
        offset_ns=offset,
        challenge=CHALLENGE,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        bank_cols=BANK_COLS,
        n_rounds=N_ROUNDS,
        jitter_sigma_ns=JITTER_SIGMA_NS,
        use_hash=False,
    )
    raw_round_bits = key_to_bits128(key_raw_round)
    raw_round_dists.append(
        hamming_distance_bits(ref_raw_round_bits, raw_round_bits)
    )

    # Scheme 3: single-shot, hashed (optional)
    key_hash_single = key_gen(
        seed=SEED,
        offset_ns=offset,
        challenge=CHALLENGE,
        n_rows=N_ROWS,
        n_cols=N_COLS,
        bank_cols=BANK_COLS,
        use_hash=True,
    )
    hash_single_bits = key_to_bits128(key_hash_single)
    hash_single_dists.append(
        hamming_distance_bits(ref_hash_single_bits, hash_single_bits)
    )


plt.figure(figsize=(8, 5))
plt.plot(offsets, raw_single_dists, label="Single-shot (raw, no hash)", lw=1.2)
plt.plot(offsets, raw_round_dists,
         label=f"Round-based (raw, R={N_ROUNDS})", lw=1.2)
plt.plot(offsets, hash_single_dists,
         label="Single-shot (hashed)", lw=1.2, linestyle="--")
plt.xlabel("Added t_set offset (ns)")
plt.ylabel("Hamming Distance from Initial Key (bits)")
plt.title("PUF Key Drift: Single-shot vs Round-based (pre- and post-hash)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("=== Drift summary ===")
print(f"Single-shot (raw):      avg={np.mean(raw_single_dists):.2f}, "
      f"max={np.max(raw_single_dists)} bits")
print(f"Round-based (raw, R={N_ROUNDS}): "
      f"avg={np.mean(raw_round_dists):.2f}, max={np.max(raw_round_dists)} bits")
print(f"Single-shot (hashed):   avg={np.mean(hash_single_dists):.2f}, "
      f"max={np.max(hash_single_dists)} bits")
print(f"Final offset: {offsets[-1]:.1f} ns")
