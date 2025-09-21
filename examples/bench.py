import time, json, csv, os
import numpy as np
import cupy as cp
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

from rbyrct_core.core import forward_project, mart_reconstruct
from rbyrct_core.fbp_baseline import cpu_fbp_baseline

def bench(N=512, n_angles=180, n_det=512, iters=10, spp=2.0, beta=1.0, out_csv="assets/bench.csv"):
    os.makedirs("assets", exist_ok=True)
    phantom = resize(shepp_logan_phantom(), (N, N), anti_aliasing=True).astype(np.float32)
    phantom /= phantom.max()
    phantom_gpu = cp.asarray(phantom)

    t0 = time.time()
    sino = forward_project(phantom_gpu, n_angles, n_det, spp=spp)
    cp.cuda.runtime.deviceSynchronize()
    fp_ms = (time.time()-t0)*1000

    t0 = time.time()
    rec_gpu = mart_reconstruct(sino, N, N, iters=iters, beta=beta, spp=spp)
    cp.cuda.runtime.deviceSynchronize()
    mart_ms = (time.time()-t0)*1000
    rec_gpu_np = cp.asnumpy(rec_gpu)

    t0 = time.time()
    rec_fbp = cpu_fbp_baseline(sino)
    cp.cuda.runtime.deviceSynchronize()
    fbp_ms = (time.time()-t0)*1000
    rec_fbp_np = cp.asnumpy(rec_fbp)

    psnr_gpu = float(peak_signal_noise_ratio(phantom, rec_gpu_np, data_range=1.0))
    ssim_gpu = float(ssim(phantom, rec_gpu_np, data_range=1.0))
    psnr_fbp = float(peak_signal_noise_ratio(phantom, rec_fbp_np, data_range=1.0))
    ssim_fbp = float(ssim(phantom, rec_fbp_np, data_range=1.0))

    row = {
        "N": N, "n_angles": n_angles, "n_det": n_det, "iters": iters, "spp": spp, "beta": beta,
        "forward_ms": round(fp_ms,1), "mart_ms": round(mart_ms,1), "fbp_ms": round(fbp_ms,1),
        "psnr_mart": round(psnr_gpu,3), "ssim_mart": round(ssim_gpu,4),
        "psnr_fbp": round(psnr_fbp,3), "ssim_fbp": round(ssim_fbp,4),
        "gpu_name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    }
    print(json.dumps(row, indent=2))

    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: w.writeheader()
        w.writerow(row)

if __name__ == "__main__":
    bench()

