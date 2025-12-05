import numpy as np
import matplotlib.pyplot as plt
from puf_gen import (
    program_traces,
    response_bits,
    compute_geometry,
    estimate_key_time_ns,
)

SEED = 0x1234
N_CHALLENGES = 500

GEOMETRIES = [
    (4, 256, 64),
    (4, 256, 128),
    (4, 256, 256),
    (4, 256, 32),
    (8, 256, 64),
    (8, 256, 128),
    (8, 256, 256),
    (16, 256, 64),
    (16, 256, 128),
    (16, 256, 256),
    (256, 256, 256)
]


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
    print("=== PUF geometry tradeoff analysis (with latency) ===")
    results = []

    for idx, (nr, nc, bc) in enumerate(GEOMETRIES, start=1):
        res = analyze_geometry(nr, nc, bc)
        results.append(res)

        print(f"Geometry #{idx}: R={res['n_rows']}, C={res['n_cols']}, BANK_COLS={res['bank_cols']}")
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


    t_array_vals = np.array([r["t_array_ns"] for r in results])
    hd_means = np.array([r["hd_mean"] for r in results])
    bit_means = np.array([r["bit_mean"] for r in results])


    print("\n=== Geometry Legend ===")
    for idx, res in enumerate(results, start=1):
        print(f"Geometry #{idx}: (R={res['n_rows']}, C={res['n_cols']}, BANK_COLS={res['bank_cols']})")


    rng = np.random.default_rng(0)
    t_std = np.std(t_array_vals) if np.std(t_array_vals) > 0 else 1.0
    hd_std = np.std(hd_means) if np.std(hd_means) > 0 else 1.0
    bm_std = np.std(bit_means) if np.std(bit_means) > 0 else 1.0

    jitter_scale = 0.01

    t_jitter_hd = t_array_vals + rng.normal(0, jitter_scale * t_std, size=len(t_array_vals))
    hd_jitter = hd_means + rng.normal(0, jitter_scale * hd_std, size=len(hd_means))

    t_jitter_bm = t_array_vals + rng.normal(0, jitter_scale * t_std, size=len(t_array_vals))
    bm_jitter = bit_means + rng.normal(0, jitter_scale * bm_std, size=len(bit_means))


    plt.figure(figsize=(8, 5))

    for idx, (t, hd) in enumerate(zip(t_jitter_hd, hd_jitter), start=1):
        plt.scatter(t, hd, s=50, label=f"Geom #{idx}")
        plt.annotate(
            f"#{idx}", (t, hd),
            textcoords="offset points", xytext=(4, 4),
            fontsize=8, fontweight="bold"
        )

    plt.xlabel("Array time per key (ns)")
    plt.ylabel("Inter-challenge HD mean (bits)")
    plt.title("Inter-challenge Hamming vs. Array Time")
    plt.grid(True, alpha=0.3)

    # Small legend anchored OUTSIDE the plot
    plt.legend(
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8, 5))

    for idx, (t, bm) in enumerate(zip(t_jitter_bm, bm_jitter), start=1):
        plt.scatter(t, bm, s=50, label=f"Geom #{idx}")
        plt.annotate(
            f"#{idx}", (t, bm),
            textcoords="offset points", xytext=(4, 4),
            fontsize=8, fontweight="bold"
        )

    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Array time per key (ns)")
    plt.ylabel("Bit mean (fraction of 1s)")
    plt.title("Bit Balance vs. Array Time")
    plt.grid(True, alpha=0.3)

    plt.legend(
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0
    )

    plt.tight_layout()
    plt.show()
