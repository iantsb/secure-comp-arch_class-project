import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log2
from puf_gen import (
    program_traces,
    response_bits,
    compute_geometry,
    estimate_key_time_ns,
)

SEED = 0x1234
N_CHALLENGES = 500

ROW_OPTIONS = [2**i for i in range(0, 9)]   # 1,2,4,...,256
COL_OPTIONS = [256]
BANK_COL_OPTIONS = [2**i for i in range(2, 9)]  # 4,8,16,...,256


def random_challenges(rng, n):
    return rng.integers(low=0, high=2**32, dtype=np.uint32, size=n)


def analyze_geometry(
    n_rows: int,
    n_cols: int,
    bank_cols: int,
    seed: int = SEED,
    n_challenges: int = N_CHALLENGES,
):
    rng = np.random.default_rng(seed)
    n_rows, n_cols, bank_cols, n_banks, pairs_per_bank, total_pairs = compute_geometry(
        n_rows, n_cols, bank_cols
    )

    tset = program_traces(seed=seed, n_rows=n_rows, n_cols=n_cols)
    chals = random_challenges(rng, n_challenges)

    all_bits = np.zeros((n_challenges, total_pairs), dtype=np.int8)

    for i, c in enumerate(chals):
        bits = response_bits(tset, int(c), n_rows, n_cols, bank_cols)
        all_bits[i, :] = bits

    overall_mean = np.mean(all_bits)

    # Inter-challenge Hamming distance (reference = first challenge)
    ref = all_bits[0, :]
    dists = np.sum(ref ^ all_bits[1:, :], axis=1)
    avg_hd = np.mean(dists)
    std_hd = np.std(dists)

    # Time estimates (single-shot)
    t_array_ns, t_total_ns = estimate_key_time_ns(
        n_rows=n_rows,
        n_cols=n_cols,
        bank_cols=bank_cols,
        t_pulse_ns=100.0,
        t_sense_logic_ns=40.0,
        hash_cycles=64,
        f_clk_MHz=200.0,
        n_rounds=1,
        include_hash=True,
    )

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "bank_cols": bank_cols,
        "n_banks": n_banks,
        "bits_per_response": total_pairs,
        "bit_mean": overall_mean,
        "hd_mean": avg_hd,
        "hd_std": std_hd,
        "t_array_ns": t_array_ns,
        "t_total_ns": t_total_ns,
    }


if __name__ == "__main__":
    print("=== PUF geometry sweep (with round-aware timing) ===")

    results = []
    geom_idx = 0

    # Generate and evaluate all geometries
    for nr in ROW_OPTIONS:
        for nc in COL_OPTIONS:
            for bc in BANK_COL_OPTIONS:
                if nc % bc != 0:
                    continue

                geom_idx += 1
                res = analyze_geometry(nr, nc, bc)
                results.append(res)

                print(
                    f"Geometry #{geom_idx}: R={res['n_rows']}, "
                    f"C={res['n_cols']}, BANK_COLS={res['bank_cols']}"
                )
                print(
                    f"   banks={res['n_banks']}, bits={res['bits_per_response']}"
                )
                print(
                    f"   bit_mean={res['bit_mean']:.3f}, "
                    f"inter-HD={res['hd_mean']:.2f}±{res['hd_std']:.2f}"
                )
                print(
                    f"   est t_array(single-round)={res['t_array_ns']:.1f} ns, "
                    f"t_total (with hash)={res['t_total_ns']:.1f} ns\n"
                )


if not results:
    print("No valid geometries found.")
    raise SystemExit(0)


# -------------------------
# Compute rounds per geometry and adjusted times
# -------------------------
for r in results:
    n_rows = r["n_rows"]
    n_banks = r["n_banks"]

    # bits required to select a row: n = log2(n_rows)
    # (ROW_OPTIONS are powers of two, so this is integral)
    if n_rows <= 1:
        # degenerate case: log2(1) == 0 -> challenge bits == 0
        # set rounds = 1 to avoid division by zero. Adjust if you'd prefer different behavior.
        n_select_bits = 0
    else:
        n_select_bits = int(round(log2(n_rows)))

    challenge_bits = n_select_bits * n_banks

    if challenge_bits <= 0:
        rounds = 1
    else:
        rounds = int(ceil(32.0 / float(challenge_bits)))

    r["n_select_bits"] = n_select_bits
    r["challenge_bits"] = challenge_bits
    r["rounds_needed"] = rounds
    r["t_array_ns_rounds"] = r["t_array_ns"] * rounds
    # Also scale t_total if you want total-with-hash multiplied by rounds:
    r["t_total_ns_rounds"] = r["t_total_ns"] * rounds


# -----------------------------------------
# Extract vectors for plotting (original metrics)
# -----------------------------------------
bank_cols_vals = np.array([r["bank_cols"] for r in results])
hd_means = np.array([r["hd_mean"] for r in results])
bit_means = np.array([r["bit_mean"] for r in results])
t_array_vals_single = np.array([r["t_array_ns"] for r in results])
t_array_vals_rounds = np.array([r["t_array_ns_rounds"] for r in results])
n_banks_vals = np.array([r["n_banks"] for r in results])
hd_stds = np.array([r["hd_std"] for r in results])

# -----------------------------------------
# Aggregate by n_banks for plotting
# -----------------------------------------
unique_banks = np.unique(n_banks_vals)
unique_banks_sorted = np.sort(unique_banks)

mean_hd_by_bank = []
std_hd_by_bank = []
mean_bit_by_bank = []
mean_tarray_by_bank_single = []
mean_tarray_by_bank_rounds = []
count_by_bank = []

for nb in unique_banks_sorted:
    mask = n_banks_vals == nb
    mean_hd_by_bank.append(np.mean(hd_means[mask]))
    # RMS of per-geometry hd stds (informational)
    std_hd_by_bank.append(np.sqrt(np.mean(hd_stds[mask] ** 2)))
    mean_bit_by_bank.append(np.mean(bit_means[mask]))
    mean_tarray_by_bank_single.append(np.mean(t_array_vals_single[mask]))
    mean_tarray_by_bank_rounds.append(np.mean(t_array_vals_rounds[mask]))
    count_by_bank.append(np.sum(mask))

mean_hd_by_bank = np.array(mean_hd_by_bank)
std_hd_by_bank = np.array(std_hd_by_bank)
mean_bit_by_bank = np.array(mean_bit_by_bank)
mean_tarray_by_bank_single = np.array(mean_tarray_by_bank_single)
mean_tarray_by_bank_rounds = np.array(mean_tarray_by_bank_rounds)
count_by_bank = np.array(count_by_bank)

# ================================================================
# PLOTTING SECTION
# ================================================================

# 1) Average inter-challenge Hamming distance (with dashed ideal at 64)
plt.figure(figsize=(7, 4))
plt.errorbar(unique_banks_sorted, mean_hd_by_bank, yerr=std_hd_by_bank, marker='o', linestyle='-')
plt.xlabel("Number of banks")
plt.ylabel("Average inter-challenge Hamming distance (bits)")
plt.title("Average inter-challenge Hamming distance vs banks")
plt.axhline(64.0, linestyle='--', linewidth=1, label='Reference = 64 bits')
plt.grid(True)
plt.xticks(unique_banks_sorted)
plt.legend()
plt.tight_layout()
plt.show()

# 2) Bit mean (dashed ideal at 0.5)
plt.figure(figsize=(7, 4))
plt.plot(unique_banks_sorted, mean_bit_by_bank, marker='o', linestyle='-')
plt.axhline(0.5, linestyle='--', linewidth=1, label='Ideal bit mean = 0.5')
plt.xlabel("Number of banks")
plt.ylabel("Average bit mean")
plt.title("Average bit mean vs banks")
plt.legend()
plt.grid(True)
plt.xticks(unique_banks_sorted)
plt.tight_layout()
plt.show()

# 3) Array time estimate: show both single-round mean and round-adjusted mean
plt.figure(figsize=(7, 4))
plt.plot(unique_banks_sorted, mean_tarray_by_bank_single, marker='o', linestyle='-', label='Single-round mean t_array (ns)')
plt.plot(unique_banks_sorted, mean_tarray_by_bank_rounds, marker='s', linestyle='--', label='Round-adjusted mean t_array (ns)')

plt.xlabel("Number of banks")
plt.ylabel("Estimated t_array (ns)")
plt.title("Estimated array time (single-round vs round-adjusted) vs banks")
plt.grid(True)
plt.xticks(unique_banks_sorted)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: save a combined figure summarizing all three subplots
try:
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].errorbar(unique_banks_sorted, mean_hd_by_bank, yerr=std_hd_by_bank, marker='o', linestyle='-')
    axes[0].axhline(64.0, linestyle='--', linewidth=1)
    axes[0].set_ylabel("Avg inter-HD (bits)")
    axes[0].grid(True)

    axes[1].plot(unique_banks_sorted, mean_bit_by_bank, marker='o', linestyle='-')
    axes[1].axhline(0.5, linestyle='--', linewidth=1)
    axes[1].set_ylabel("Avg bit mean")
    axes[1].grid(True)

    axes[2].plot(unique_banks_sorted, mean_tarray_by_bank_single, marker='o', linestyle='-', label='single-round')
    axes[2].plot(unique_banks_sorted, mean_tarray_by_bank_rounds, marker='s', linestyle='--', label='round-adjusted')
    axes[2].set_xlabel("Number of banks")
    axes[2].set_ylabel("t_array (ns)")
    axes[2].grid(True)
    axes[2].legend()

    plt.xticks(unique_banks_sorted)
    plt.tight_layout()
    fig.savefig("puf_banks_summary_rounds.png", dpi=150)
    print("Saved combined summary figure to puf_banks_summary_rounds.png")
except Exception as e:
    print(f"Could not save combined figure: {e}")
