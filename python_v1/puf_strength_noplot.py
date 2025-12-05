import numpy as np
from puf_gen import (
    program_traces,
    response_bits,
    compute_geometry,
    estimate_key_time_ns,
)

SEED = 0x1234
N_CHALLENGES = 500


ROW_OPTIONS = [2**i for i in range(0, 9)]   # 1,2,4,...,256
COL_OPTIONS = [256, 512]
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

    # Bit balance
    overall_mean = np.mean(all_bits)

    # Inter-challenge Hamming distances (vs first challenge)
    ref = all_bits[0, :]
    dists = np.sum(ref ^ all_bits[1:, :], axis=1)
    avg_hd = np.mean(dists)
    std_hd = np.std(dists)

    # Latency estimates (single-shot, with hash)
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
    print("=== PUF geometry sweep (no plotting) ===")

    results = []
    geom_idx = 0

    # Generate and evaluate all geometries
    for nr in ROW_OPTIONS:
        for nc in COL_OPTIONS:
            for bc in BANK_COL_OPTIONS:
                # BANK_COLS must divide N_COLS
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
                    f"   est t_array={res['t_array_ns']:.1f} ns, "
                    f"t_total (with hash)={res['t_total_ns']:.1f} ns"
                )
                print()


if not results:
    print("No valid geometries found with the given constraints.")
    raise SystemExit(0)


hd_means = np.array([r["hd_mean"] for r in results])
bit_means = np.array([r["bit_mean"] for r in results])
bits_counts = np.array([r["bits_per_response"] for r in results])
t_array_vals = np.array([r["t_array_ns"] for r in results])

# Ideal values
ideal_hd = bits_counts / 2.0
ideal_bit = 0.5


hd_err = np.abs(hd_means - ideal_hd)
bit_err = np.abs(bit_means - ideal_bit)


best_hd_idx_overall = int(np.argmin(hd_err))
best_bit_idx_overall = int(np.argmin(bit_err))

print("\n ideal geometries w/o time limit")

bh = results[best_hd_idx_overall]
print("\nBest overall inter-HD (closest to bits/2):")
print(
    f"  R={bh['n_rows']}, C={bh['n_cols']}, BANK_COLS={bh['bank_cols']}, "
    f"banks={bh['n_banks']}, bits={bh['bits_per_response']}"
)
print(
    f"  hd_mean={bh['hd_mean']:.2f}, ideal={bh['bits_per_response'] / 2:.2f}, "
    f"|error|={hd_err[best_hd_idx_overall]:.2f}"
)
print(
    f"  bit_mean={bh['bit_mean']:.3f}, "
    f"t_array={bh['t_array_ns']:.1f} ns, t_total={bh['t_total_ns']:.1f} ns"
)

bb = results[best_bit_idx_overall]
print("\nBest overall bit_mean (closest to 0.5):")
print(
    f"  R={bb['n_rows']}, C={bb['n_cols']}, BANK_COLS={bb['bank_cols']}, "
    f"banks={bb['n_banks']}, bits={bb['bits_per_response']}"
)
print(
    f"  bit_mean={bb['bit_mean']:.3f}, ideal=0.500, "
    f"|error|={bit_err[best_bit_idx_overall]:.3f}"
)
print(
    f"  hd_mean={bb['hd_mean']:.2f}, ideal={bb['bits_per_response'] / 2:.2f}, "
    f"|error|={hd_err[best_bit_idx_overall]:.2f}"
)
print(
    f"  t_array={bb['t_array_ns']:.1f} ns, t_total={bb['t_total_ns']:.1f} ns"
)

# -----------------------------
# 2. Best geometries WITH TIME LIMIT (t_array <= 1000 ns)
# -----------------------------
TIME_LIMIT_NS = 1000.0

valid_mask = t_array_vals <= TIME_LIMIT_NS

print("\n ideal geometry w/time constraint (t_array <= 1000 ns) ")

if not np.any(valid_mask):
    print("No geometries meet the time constraint.")
else:
    # Extract only geometries under the time limit
    hd_err_limited = hd_err[valid_mask]
    bit_err_limited = bit_err[valid_mask]

    # Indexing back into results array
    limited_indices = np.where(valid_mask)[0]

    # Best HD under time limit
    best_hd_idx_lim_local = int(np.argmin(hd_err_limited))
    best_hd_idx_lim = limited_indices[best_hd_idx_lim_local]
    bhr = results[best_hd_idx_lim]

    print("\nBest inter-HD under time constraint:")
    print(
        f"  R={bhr['n_rows']}, C={bhr['n_cols']}, BANK_COLS={bhr['bank_cols']}, "
        f"banks={bhr['n_banks']}, bits={bhr['bits_per_response']}"
    )
    print(
        f"  hd_mean={bhr['hd_mean']:.2f}, ideal={bhr['bits_per_response'] / 2:.2f}, "
        f"|error|={hd_err[best_hd_idx_lim]:.2f}"
    )
    print(
        f"  bit_mean={bhr['bit_mean']:.3f}, "
        f"t_array={bhr['t_array_ns']:.1f} ns, t_total={bhr['t_total_ns']:.1f} ns"
    )

    # Best bit_mean under time limit
    best_bit_idx_lim_local = int(np.argmin(bit_err_limited))
    best_bit_idx_lim = limited_indices[best_bit_idx_lim_local]
    bbr = results[best_bit_idx_lim]

    print("\nBest bit_mean under time constraint:")
    print(
        f"  R={bbr['n_rows']}, C={bbr['n_cols']}, BANK_COLS={bbr['bank_cols']}, "
        f"banks={bbr['n_banks']}, bits={bbr['bits_per_response']}"
    )
    print(
        f"  bit_mean={bbr['bit_mean']:.3f}, ideal=0.500, "
        f"|error|={bit_err[best_bit_idx_lim]:.3f}"
    )
    print(
        f"  hd_mean={bbr['hd_mean']:.2f}, ideal={bbr['bits_per_response'] / 2:.2f}, "
        f"|error|={hd_err[best_bit_idx_lim]:.2f}"
    )
    print(
        f"  t_array={bbr['t_array_ns']:.1f} ns, t_total={bbr['t_total_ns']:.1f} ns"
    )
