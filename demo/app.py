# app.py
import time
import numpy as np
import cupy as cp
import streamlit as st
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim

from rbyrct_core import forward_project, mart_reconstruct, cpu_fbp_baseline
from rbyrct_core.core import forward_project, mart_reconstruct
from rbyrct_core.fbp_baseline import cpu_fbp_baseline


st.set_page_config(page_title="CUDA-Accelerated RBYRCT (MART) Demo", layout="wide")
st.title("CUDA-Accelerated RBYRCT (MART) — Golden Ticket Demo")

# Sidebar controls
N = st.sidebar.slider("Image size (N x N)", 128, 1024, 512, step=64)
n_angles = st.sidebar.slider("Projections (angles)", 30, 720, 180, step=10)
n_det = st.sidebar.slider("Detector bins", 64, 1024, 512, step=64)
iters = st.sidebar.slider("MART iterations", 1, 50, 10)
beta = st.sidebar.slider("MART beta", 0.1, 2.0, 1.0, 0.1)
spp = st.sidebar.slider("Samples-per-pixel along ray", 1.0, 4.0, 2.0, 0.5)
det_spacing = 1.0

col1, col2, col3 = st.columns(3)

# Phantom
phantom = resize(shepp_logan_phantom(), (N, N), anti_aliasing=True).astype(np.float32)
phantom = phantom / phantom.max()

with col1:
    st.subheader("Ground Truth")
    st.image(phantom, clamp=True, use_container_width=True)

# Move to GPU
phantom_gpu = cp.asarray(phantom)

# Generate synthetic sinogram (parallel beam)
t0 = time.time()
sino_gpu = forward_project(phantom_gpu, n_angles, n_det, det_spacing=det_spacing, spp=spp)
cp.cuda.runtime.deviceSynchronize()
gpu_fp_ms = (time.time()-t0)*1000

with col2:
    st.subheader("Sinogram (GPU forward)")
    st.caption(f"Forward-project time: {gpu_fp_ms:.1f} ms")
    st.image(cp.asnumpy(sino_gpu), clamp=True, use_container_width=True)

# GPU MART recon
t0 = time.time()
rec_gpu = mart_reconstruct(sino_gpu, N, N, iters=iters, beta=beta, det_spacing=det_spacing, spp=spp)
cp.cuda.runtime.deviceSynchronize()
gpu_ms = (time.time()-t0)*1000
rec_gpu_np = cp.asnumpy(rec_gpu)
psnr_gpu = peak_signal_noise_ratio(phantom, rec_gpu_np, data_range=1.0)
ssim_gpu = ssim(phantom, rec_gpu_np, data_range=1.0)

# CPU FBP baseline (very rough; for speed contrast)
t0 = time.time()
rec_fbp_gpu = cpu_fbp_baseline(sino_gpu)
cp.cuda.runtime.deviceSynchronize()
cpu_ms = (time.time()-t0)*1000
rec_fbp_np = cp.asnumpy(rec_fbp_gpu)
psnr_fbp = peak_signal_noise_ratio(phantom, rec_fbp_np, data_range=1.0)
ssim_fbp = ssim(phantom, rec_fbp_np, data_range=1.0)

with col3:
    st.subheader("Reconstruction (GPU MART)")
    st.caption(f"GPU MART: {gpu_ms:.1f} ms for {iters} iters")
    st.image(rec_gpu_np, clamp=True, use_container_width=True)

st.markdown("### Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("PSNR (GPU MART)", f"{psnr_gpu:.2f} dB")
m2.metric("SSIM (GPU MART)", f"{ssim_gpu:.3f}")
m3.metric("Speedup (vs CPU FBP*)", f"{(cpu_ms/max(gpu_ms,1e-3)):.1f}×")
st.caption("*FBP here is computed with scikit-image’s CPU iradon; shown for rough baseline, not apples-to-apples.")

st.markdown("### Side-by-Side")
c1, c2 = st.columns(2)
with c1:
    st.image(rec_fbp_np, caption=f"CPU FBP (PSNR {psnr_fbp:.2f}, SSIM {ssim_fbp:.3f})", clamp=True, use_container_width=True)
with c2:
    st.image(rec_gpu_np, caption=f"GPU MART (PSNR {psnr_gpu:.2f}, SSIM {ssim_gpu:.3f})", clamp=True, use_container_width=True)

st.markdown("---")
st.markdown("Built with **CUDA via CuPy RawKernels**, profiled with Nsight (recommended), and packaged for a contest-ready OSS demo.")

