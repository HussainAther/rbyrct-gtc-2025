import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM

# Assuming rbyrct_core is available
from rbyrct_core import forward_project, mart_reconstruct, cpu_fbp_baseline

GPU = False
try:
    import cupy as cp
    GPU = True
    print("✅ CuPy imported. Running in GPU mode.")
except ImportError:
    # Define cp as np for minimal changes in array creation, but rely on 'GPU' flag
    # to skip CuPy-specific functions like cp.asnumpy().
    cp = np 
    print("⚠️ CuPy not found, falling back to NumPy (CPU-only mode).")


# Ensure assets folder exists
os.makedirs("assets", exist_ok=True)
# Config
sizes = [64, 128, 256]
n_angles = 90
n_det = 128
iters = 10
beta = 1.0
det_spacing, spp = 1.0, 1.0

records = []

for H in sizes:
    W = H
    
    # Use 'cp' for array creation. If in CPU mode, 'cp' is 'np'.
    phantom = cp.zeros((H, W), dtype=cp.float32)
    phantom[H//4:H//2, W//4:W//2] = 1.0

    # CORRECTED LOGIC for the 'asnumpy' error: Only use cp.asnumpy() if CuPy (GPU) is active.
    if GPU:
        phantom_np = cp.asnumpy(phantom)
    else:
        phantom_np = phantom

    # --- Forward projection ---
    # forward_project should handle the type of array it receives (np or cp)
    t0 = time.time()
    sino = forward_project(phantom, n_angles=n_angles, n_det=n_det,
                           det_spacing=det_spacing, spp=spp)
    fp_ms = (time.time() - t0) * 1000

    # --- MART Reconstruction (Conditional) ---
    mart_ms = np.nan
    psnr_mart = np.nan
    ssim_mart = np.nan
    rec_mart_np = np.zeros_like(phantom_np) # Placeholder

    # MART is typically GPU-accelerated. Only run if CuPy is available.
    if GPU:
        t0 = time.time()
        rec_mart = mart_reconstruct(sino, H=H, W=W, iters=iters, beta=beta,
                                    det_spacing=det_spacing, spp=spp)
        mart_ms = (time.time() - t0) * 1000
        rec_mart_np = cp.asnumpy(rec_mart)

        # Metrics for MART
        psnr_mart = PSNR(phantom_np, rec_mart_np, data_range=1.0)
        ssim_mart = SSIM(phantom_np, rec_mart_np, data_range=1.0)
    else:
        print(f"Skipping MART for size {H} (CPU-only mode).")

    # --- FBP (CPU baseline) ---
    # Convert sinogram to NumPy array *only* if we are in GPU mode.
    if GPU:
        sino_np = cp.asnumpy(sino)
    else:
        sino_np = sino
        
    t0 = time.time()
    # CORRECTED LOGIC for the 'ValueError: Input images must have the same dimensions.'
    # Pass the target output size H to ensure rec_fbp matches phantom_np's shape (H, H).
    rec_fbp = cpu_fbp_baseline(sino_np, output_size=H) 
    fbp_ms = (time.time() - t0) * 1000

    # --- Metrics for FBP ---
    psnr_fbp  = PSNR(phantom_np, rec_fbp, data_range=1.0)
    ssim_fbp  = SSIM(phantom_np, rec_fbp, data_range=1.0)

    records.append(dict(
        size=H,
        forward_ms=fp_ms,
        mart_ms=mart_ms,
        fbp_ms=fbp_ms,
        psnr_mart=psnr_mart,
        psnr_fbp=psnr_fbp,
        ssim_mart=ssim_mart,
        ssim_fbp=ssim_fbp,
    ))

# Save raw CSV
df = pd.DataFrame(records)
df.to_csv("assets/bench_grid.csv", index=False)

# Save Markdown summary
with open("assets/bench_table.md", "w") as f:
    f.write("| Size | Forward (ms) | MART (ms) | FBP (ms) | PSNR MART | PSNR FBP | SSIM MART | SSIM FBP |\n")
    f.write("|------|--------------|-----------|----------|-----------|----------|-----------|----------|\n")
    for row in records:
        # Use '-' for NaN values in the table if MART was skipped
        mart_ms_str = f"{row['mart_ms']:.1f}" if not np.isnan(row['mart_ms']) else "-"
        psnr_mart_str = f"{row['psnr_mart']:.2f}" if not np.isnan(row['psnr_mart']) else "-"
        ssim_mart_str = f"{row['ssim_mart']:.3f}" if not np.isnan(row['ssim_mart']) else "-"
        
        f.write(f"| {row['size']} | {row['forward_ms']:.1f} | {mart_ms_str} "
                f"| {row['fbp_ms']:.1f} | {psnr_mart_str} | {row['psnr_fbp']:.2f} "
                f"| {ssim_mart_str} | {row['ssim_fbp']:.3f} |\n")

# Runtime scaling plot
plt.figure(figsize=(6,4))
# Only plot MART if it was run (i.e., if GPU is True)
if GPU:
    plt.plot(df["size"], df["mart_ms"], "-o", label="MART (GPU)")
plt.plot(df["size"], df["fbp_ms"], "-o", label="FBP (CPU)")
plt.xlabel("Phantom size (pixels)")
plt.ylabel("Runtime (ms)")
plt.title("Runtime scaling")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("assets/runtime_vs_size.png", dpi=150)
plt.close()

print("✅ Saved: assets/bench_grid.csv, assets/bench_table.md, assets/runtime_vs_size.png")
