import time
import numpy as np

try:
    import cupy as cp
    GPU = True
except ImportError:
    cp = np  # fallback to NumPy
    GPU = False

from rbyrct_core import mart_reconstruct, cpu_fbp_baseline


def main():
    H = W = 128
    n_angles = 60
    n_det = 128

    # phantom image
    x = np.zeros((H, W), dtype=np.float32)
    x[H//4:H//2, W//4:W//2] = 1.0

    # simple sinogram: projections = sum along one axis
    sino = np.sum(x, axis=0, keepdims=True).repeat(n_angles, axis=0)

    # MART reconstruction
    t0 = time.time()
    rec_mart = mart_reconstruct(sino, H=H, W=W, iters=5, beta=1.0)
    t1 = time.time()

    # CPU FBP baseline
    rec_fbp = cpu_fbp_baseline(sino, output_size=H)
    t2 = time.time()

    print("=== Benchmark ===")
    print("GPU available?", GPU)
    print(f"MART time: {t1 - t0:.3f} s")
    print(f"FBP  time: {t2 - t1:.3f} s")
    print("MART shape:", rec_mart.shape)
    print("FBP  shape:", rec_fbp.shape)


if __name__ == "__main__":
    main()

