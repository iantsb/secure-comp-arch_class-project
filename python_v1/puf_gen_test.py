# puf_stability_analysis.py
# Performs key stability test by gradually increasing t_set offsets.

import numpy as np
import matplotlib.pyplot as plt
from puf_gen import key_gen


SEED = 0x16FB
CHALLENGE = 0x7D924EC2
NUM_RUNS = 1000
OFFSET_STEP = 0.1  # ns per iteration

key1 = key_gen(seed=SEED, offset_ns=0.0, challenge=CHALLENGE)
ref_key = np.frombuffer(key1, dtype=np.uint8)


hamming_distances = []

for i in range(1, NUM_RUNS + 1):
    offset = i * OFFSET_STEP
    key=key_gen(SEED, offset, CHALLENGE)
    key_bytes = np.frombuffer(key, dtype=np.uint8)

    # hamming distance
    xor = np.unpackbits(ref_key ^ key_bytes)
    hamming = np.sum(xor)
    hamming_distances.append(hamming)


offsets = np.arange(1, NUM_RUNS + 1) * OFFSET_STEP

plt.figure(figsize=(8, 5))
plt.plot(offsets, hamming_distances, lw=1.5)
plt.xlabel("Added t_set offset (ns)")
plt.ylabel("Hamming Distance from Initial Key (bits)")
plt.title("PUF Key Drift over Global t_set Offset")
plt.grid(True, alpha=0.3)
plt.show()


print(f"Average Hamming distance: {np.mean(hamming_distances):.2f} bits")
print(f"Max distance: {np.max(hamming_distances)} bits")
print(f"Final offset: {offsets[-1]:.1f} ns")
