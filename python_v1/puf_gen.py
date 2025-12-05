import numpy as np
import hashlib


DEFAULT_N_ROWS = 4
DEFAULT_N_COLS = 256
DEFAULT_BANK_COLS = 16

def compute_geometry(n_rows, n_cols, bank_cols):
    assert n_cols % bank_cols == 0, "BANK_COLS must divide N_COLS"
    assert bank_cols % 2 == 0, "BANK_COLS must be even (pairs of columns)"
    n_banks = n_cols // bank_cols
    pairs_per_bank = bank_cols // 2
    total_pairs = n_banks * pairs_per_bank
    return n_rows, n_cols, bank_cols, n_banks, pairs_per_bank, total_pairs



def program_traces(seed=1234,
                   mean_ns=100.0,
                   sigma_ns=15.0,
                   offset_ns=0.0,
                   n_rows=DEFAULT_N_ROWS,
                   n_cols=DEFAULT_N_COLS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t_set = rng.normal(loc=mean_ns, scale=sigma_ns, size=(n_rows, n_cols))
    t_set = np.clip(t_set + offset_ns, 60.0, 160.0)
    return t_set



def decode_bank_row(chal_32: int, bank_id: int) -> int:
    # 2 challenge bits per bank
    shift = 2 * bank_id
    return (chal_32 >> shift) & 0b11  # 0..3


def race_bit(tset: np.ndarray, row: int, left_col: int) -> int:
    c = left_col
    tL = tset[row, c]
    tR = tset[row, c + 1]
    return int(tL < tR)


def bank_response_bits(tset: np.ndarray,
                       row: int,
                       base_col: int,
                       bank_cols: int) -> list[int]:
    bits = []
    for i in range(0, bank_cols, 2):
        bits.append(race_bit(tset, row, base_col + i))
    return bits


def response_bits(tset: np.ndarray,
                  chal_32: int,
                  n_rows: int = DEFAULT_N_ROWS,
                  n_cols: int = DEFAULT_N_COLS,
                  bank_cols: int = DEFAULT_BANK_COLS) -> list[int]:

    n_rows, n_cols, bank_cols, n_banks, pairs_per_bank, total_pairs = compute_geometry(
        n_rows, n_cols, bank_cols
    )
    bits = []
    for b in range(n_banks):
        row_sel = decode_bank_row(chal_32, b) % n_rows
        base_col = b * bank_cols
        bits += bank_response_bits(tset, row_sel, base_col, bank_cols)
    assert len(bits) == total_pairs
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    assert len(bits) % 8 == 0
    byts = bytearray(len(bits) // 8)
    for i, b in enumerate(bits):
        if b:
            byts[i // 8] |= 1 << (7 - (i % 8))
    return bytes(byts)


def key_from_bits_hash(bits: list[int]) -> bytes:

    raw = bits_to_bytes(bits)
    digest = hashlib.sha256(raw).digest()
    return digest[:16]  # 128-bit key


def key_from_bits_nohash(bits: list[int]) -> bytes:

    return bits_to_bytes(bits)


#
def round_based_bits(tset_base: np.ndarray,
                     chal_32: int,
                     n_rounds: int = 3,
                     jitter_sigma_ns: float = 0.5,
                     n_rows: int = DEFAULT_N_ROWS,
                     n_cols: int = DEFAULT_N_COLS,
                     bank_cols: int = DEFAULT_BANK_COLS,
                     seed: int = None) -> list[int]:

    rng = np.random.default_rng(seed)
    n_rows, n_cols, bank_cols, n_banks, pairs_per_bank, total_pairs = compute_geometry(
        n_rows, n_cols, bank_cols
    )

    all_round_bits = np.zeros((n_rounds, total_pairs), dtype=np.int8)

    for r in range(n_rounds):
        # Add per-read noise (keeps global offset intact)
        jitter = rng.normal(loc=0.0, scale=jitter_sigma_ns, size=tset_base.shape)
        tset_round = tset_base + jitter

        bits_r = response_bits(tset_round, chal_32, n_rows, n_cols, bank_cols)
        all_round_bits[r, :] = bits_r

    # Majority vote across rounds
    votes = np.sum(all_round_bits, axis=0)  # number of 1's per bit
    majority_bits = (votes >= (n_rounds // 2 + 1)).astype(int).tolist()
    return majority_bits



def key_gen(seed: int = 1234,
           offset_ns: float = 0.0,
           challenge: int = 0x1234ABCD,
           n_rows: int = DEFAULT_N_ROWS,
           n_cols: int = DEFAULT_N_COLS,
           bank_cols: int = DEFAULT_BANK_COLS,
           use_hash: bool = True) -> bytes:

    tset = program_traces(seed=seed,
                          offset_ns=offset_ns,
                          n_rows=n_rows,
                          n_cols=n_cols)
    bits = response_bits(tset, challenge, n_rows, n_cols, bank_cols)

    if use_hash:
        return key_from_bits_hash(bits)
    else:
        return key_from_bits_nohash(bits)


def key_gen_round_based(seed: int = 1234,
                        offset_ns: float = 0.0,
                        challenge: int = 0x1234ABCD,
                        n_rows: int = DEFAULT_N_ROWS,
                        n_cols: int = DEFAULT_N_COLS,
                        bank_cols: int = DEFAULT_BANK_COLS,
                        n_rounds: int = 3,
                        jitter_sigma_ns: float = 0.5,
                        use_hash: bool = True) -> bytes:

    tset_base = program_traces(seed=seed,
                               offset_ns=offset_ns,
                               n_rows=n_rows,
                               n_cols=n_cols)
    bits = round_based_bits(
        tset_base,
        challenge,
        n_rounds=n_rounds,
        jitter_sigma_ns=jitter_sigma_ns,
        n_rows=n_rows,
        n_cols=n_cols,
        bank_cols=bank_cols,
        seed=seed ^ 0xDEADBEEF,
    )

    if use_hash:
        return key_from_bits_hash(bits)
    else:
        return key_from_bits_nohash(bits)


from typing import Tuple

def estimate_key_time_ns(
    n_rows: int,
    n_cols: int,
    bank_cols: int,
    t_pulse_ns: float = 100.0,
    t_sense_logic_ns: float = 40.0,
    hash_cycles: int = 64,
    f_clk_MHz: float = 200.0,
    n_rounds: int = 1,
    include_hash: bool = True,
) -> Tuple[float, float]:

    n_rows, n_cols, bank_cols, n_banks, pairs_per_bank, total_pairs = compute_geometry(
        n_rows, n_cols, bank_cols
    )

    # Banks are always sequential → one access per bank per round
    accesses = n_banks
    t_array_ns = n_rounds * accesses * (t_pulse_ns + t_sense_logic_ns)

    # hashing (e.g., SHA-256)
    t_total_ns = t_array_ns
    if include_hash:
        t_cycle_ns = 1e3 / f_clk_MHz  # MHz -> ns
        t_hash_ns = hash_cycles * t_cycle_ns
        t_total_ns += t_hash_ns

    return t_array_ns, t_total_ns



if __name__ == "__main__":

    tset = program_traces(seed=0x9E4B, n_rows=DEFAULT_N_ROWS, n_cols=DEFAULT_N_COLS)
    chal = 0x5C17A6D3

    bits = response_bits(tset, chal)
    key_h = key_gen(seed=0x9E4B, offset_ns=0.0, challenge=chal, use_hash=True)
    key_r = key_gen_round_based(seed=0x9E4B, offset_ns=0.0, challenge=chal,
                                n_rounds=5, use_hash=True)

    print(f"32-bit challenge = 0x{chal:08X}")
    print(f"raw response vector (len={len(bits)}): {''.join(map(str,bits))}")
    print(f"single-shot hashed key: {key_h.hex()}")
    print(f"round-based hashed key: {key_r.hex()}")
    print(f"bit balance: ones={sum(bits)}, zeros={len(bits) - sum(bits)}")
