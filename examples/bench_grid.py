# examples/bench_grid.py
import os, time, csv, math, json
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
from skimage.transform import iradon

# project imports (works whether installed or run via PYTHONPATH)
from rbyrct_core import forward_project, mart_reconstruct, GPU

# ---------- config ----------
SIZES   = [128, 256, 512]     # image size N x N
ANGLES  = [90, 180]           # projections
ITERS   = [5, 10]             # MART iterations
DET_PER = 1.0                 # detector bins per pixel (n_det = int(N*DET_PER))
BETA    = 1.0
DET_SP  = 1.0
SPP     = 1.5                 # samples-per-pixel along rays
ASSETS  = "assets"
CSV_OUT = os.path.join(ASSETS, "bench_grid.csv")
MD_OUT  = os.path.join(ASSETS, "bench_table.md")

os.makedirs(ASSETS, exist_ok=True)

def make_phantom(N: int) -> np.ndarray:
    """Simple square phantom in [0,1] range."""
    x = np.zeros((N, N), dtype=np.float32)
    a, b = N//4, N//2
    x[a:b, a:b] = 1.0
    return x

def run_case(N, n_angles, iters):
    """Run FP + MART + FBP for one configuration; return metrics/timings."""
    # phantom
    ph_np = make_phantom(N)
    try:
        import cupy as cp
        ph = cp.asarray(ph_np) if GPU else ph_np
    except Exception:
        cp = None
        ph = ph_np

    n_det = int(N * DET_PER)
    # ---- forward projection ----
    t0 = time.time()
    sino = forward_project(ph, n_angles=n_angles, n_det=n_det, det_spacing=DET_SP, spp=SPP)
    fp_ms = (time.time() - t0) * 1000.0

    # ---- MART ----
    t0 = time.time()
    rec_mart = mart_reconstruct(sino, H=N, W=N, iters=iters, beta=BETA, det_spacing=DET_SP, spp=SPP)
    mart_ms = (time.time() - t0) * 1000.0

    # move arrays to NumPy for metrics + FBP input
    if GPU and cp is not None:
        sino_np = cp.asnumpy(sino)
        rec_mart_np = cp.asnumpy(rec_mart)
    else:
        sino_np = np.asarray(sino)
        rec_mart_np = np.asarray(rec_mart)

    # ---- FBP (CPU) with explicit theta/output size) ----
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    t0 = time.time()
    rec_fbp = iradon(sino_np.T, theta=theta, circle=True, output_size=N).astype(np.float32)
    fbp_ms = (time.time() - t0) * 1000.0

    # ---- metrics ----
    psnr_mart = float(PSNR(ph_np, rec_mart_np, data_range=1.0))
    psnr_fbp  = float(PSNR(ph_np, rec_fbp,     data_range=1.0))
    ssim_mart = float(SSIM(ph_np, rec_mart_np, data_range=1.0))
    ssim_fbp  = float(SSIM(ph_np, rec_fbp,     data_range=1.0))
    speedup   = fbp_ms / mart_ms if mart_ms > 0 else math.inf

    return {
        "N": N, "angles": n_angles, "iters": iters, "det": n_det,
        "fp_ms": fp_ms, "mart_ms": mart_ms, "fbp_ms": fbp_ms, "speedup": speedup,
        "psnr_mart": psnr_mart, "psnr_fbp": psnr_fbp,
        "ssim_mart": ssim_mart, "ssim_fbp": ssim_fbp,
        "gpu": bool(GPU),
    }

def main():
    rows = []
    for N in SIZES:
        for n_angles in ANGLES:
            for iters in ITERS:
                res = run_case(N, n_angles, iters)
                rows.append(res)
                print(
                    f"[N={N} angles={n_angles} iters={iters}] "
                    f"FBP {res['fbp_ms']:.1f} ms | MART {res['mart_ms']:.1f} ms "
                    f"| speedup {res['speedup']:.1f}× | "
                    f"PSNR (M/F) {res['psnr_mart']:.2f}/{res['psnr_fbp']:.2f} "
                    f"SSIM (M/F) {res['ssim_mart']:.3f}/{res['ssim_fbp']:.3f}"
                )

    # ---- save CSV ----
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "N","angles","iters","det","fp_ms","mart_ms","fbp_ms","speedup",
            "psnr_mart","psnr_fbp","ssim_mart","ssim_fbp","gpu"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Saved CSV: {CSV_OUT}")

    # ---- make a compact markdown table (first row for each N with angles=180, iters=10 if present) ----
    focus = []
    for N in SIZES:
        best = [r for r in rows if r["N"]==N and r["angles"]==180 and r["iters"]==10]
        if not best and rows:
            # fallback: first row for this N
            best = [next(r for r in rows if r["N"]==N)]
        if best:
            r = best[0]
            focus.append(r)

    md_lines = []
    md_lines.append("| Setting | CPU FBP (ms) | MART (ms) | Speedup | PSNR M/F (dB) | SSIM M/F |")
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for r in focus:
        md_lines.append(
            f"| {r['N']}² / {r['angles']} proj / det={r['det']} / {r['iters']} it "
            f"| {r['fbp_ms']:.1f} | {r['mart_ms']:.1f} | {r['speedup']:.1f}× "
            f"| {r['psnr_mart']:.2f}/{r['psnr_fbp']:.2f} | {r['ssim_mart']:.3f}/{r['ssim_fbp']:.3f} |"
        )
    with open(MD_OUT, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"✅ Saved Markdown table: {MD_OUT}")

    # ---- plot: runtime vs image size (angles=180, iters=10) ----
    try:
        import pandas as pd
        df = pd.read_csv(CSV_OUT)
        dff = df[(df.angles==180) & (df.iters==10)]
        dff = dff.sort_values("N")
        plt.figure(figsize=(6,4))
        plt.plot(dff["N"], dff["fbp_ms"], marker="o", label="CPU FBP")
        plt.plot(dff["N"], dff["mart_ms"], marker="o", label=("GPU MART" if GPU else "MART"))
        plt.xlabel("Image size (N)"); plt.ylabel("Time (ms)")
        plt.title("Runtime vs Image Size (angles=180, iters=10)")
        plt.legend()
        plt.tight_layout()
        png_path = os.path.join(ASSETS, "runtime_vs_size.png")
        plt.savefig(png_path, dpi=150)
        print(f"✅ Saved plot: {png_path}")
    except Exception as e:
        print("Plotting skipped:", e)

    # print a README-ready table to console
    print("\nREADME paste:\n")
    print("\n".join(md_lines))

if __name__ == "__main__":
    main()

