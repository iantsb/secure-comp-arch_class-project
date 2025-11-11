import numpy as np
from typing import List, Tuple, Dict, Any

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


def encrypt_placeholder(chunk: List[float]) -> List[float]:
    # encryption scheme (no-op for now)
    return chunk


def decrypt_placeholder(chunk: List[float]) -> List[float]:
    # decryption scheme (no-op for now)
    return chunk


def transmit_over_bus(data: List[float]) -> Tuple[List[float], List[List[float]]]:
    transmitted: List[float] = []
    bus_log: List[List[float]] = []
    for i in range(0, len(data), TX_SIZE):
        chunk = data[i:i + TX_SIZE]
        enc = encrypt_placeholder(chunk)
        bus_log.append(enc)
        dec = decrypt_placeholder(enc)
        transmitted.extend(dec)
    return transmitted, bus_log


if __name__ == "__main__":
    # fake MNIST weights
    weights, layout = create_small_mnist_weights(seed=123)
    print(f"total values generated: {len(weights)}")

    # transmission (unpack both returned values)
    received, bus_log = transmit_over_bus(weights)
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
