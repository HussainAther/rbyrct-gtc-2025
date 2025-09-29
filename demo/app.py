# demo/app.py
import streamlit as st
import numpy as np
import cupy as cp
import time
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
from skimage.transform import iradon

from rbyrct_core import forward_project, mart_reconstruct

st.set_page_config(page_title="RBYRCT CT Demo", page_icon="🩻", layout="wide")

st.title("🩻 RBYRCT: CUDA-Accelerated CT Reconstruction")
st.markdown(
    """
    Compare **FBP (CPU)** vs **MART (GPU)** on a simple CT phantom.  
    - Adjust scan + reconstruction parameters in the sidebar.  
    - Metrics (PSNR/SSIM) update in real time.  
    """
)

# Sidebar
st.sidebar.header("Scan Parameters")
H = W = st.sidebar.slider("Image size (pixels)", 64, 512, 256, step=64)
n_angles = st.sidebar.slider("Number of angles", 30, 360, 180, step=10)
n_det = st.sidebar.slider("Detector bins", 64, 512, 256, step=64)
iters = st.sidebar.slider("MART iterations", 1, 50, 10)
beta = st.sidebar.slider("Relaxation (beta)", 0.1, 2.0, 1.0, step=0.1)
det_spacing, spp = 1.0, 1.5

# Phantom
phantom = cp.zeros((H, W), dtype=cp.float32)
phantom[H//4:H//2, W//4:W//2] = 1.0
phantom_np = cp.asnumpy(phantom)

# Forward projection
t0 = time.time()
sino = forward_project(phantom, n_angles=n_angles, n_det=n_det,
                       det_spacing=det_spacing, spp=spp)
fp_ms = (time.time() - t0) * 1000

# MART (GPU)
t0 = time.time()
rec_mart = mart_reconstruct(sino, H=H, W=W, iters=iters, beta=beta,
                            det_spacing=det_spacing, spp=spp)
mart_ms = (time.time() - t0) * 1000
rec_mart_np = cp.asnumpy(rec_mart)

# FBP (CPU baseline with explicit output size)
sino_np = cp.asnumpy(sino)
theta = np.linspace(0., 180., n_angles, endpoint=False)
t0 = time.time()
rec_fbp = iradon(sino_np.T, theta=theta, circle=True, output_size=H).astype(np.float32)
fbp_ms = (time.time() - t0) * 1000

# Metrics
psnr_mart = PSNR(phantom_np, rec_mart_np, data_range=1.0)
psnr_fbp  = PSNR(phantom_np, rec_fbp,     data_range=1.0)
ssim_mart = SSIM(phantom_np, rec_mart_np, data_range=1.0)
ssim_fbp  = SSIM(phantom_np, rec_fbp,     data_range=1.0)

# Display
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Ground Truth")
    st.image(phantom_np, clamp=True, caption=f"Phantom ({H}×{W})", use_container_width=True)
with col2:
    st.subheader("FBP (CPU)")
    st.image(rec_fbp, clamp=True, caption=f"{fbp_ms:.1f} ms", use_container_width=True)
with col3:
    st.subheader("MART (GPU)")
    st.image(rec_mart_np, clamp=True, caption=f"{mart_ms:.1f} ms", use_container_width=True)

st.markdown("### 📊 Metrics")
st.write(f"**PSNR (dB)** → MART: {psnr_mart:.2f}, FBP: {psnr_fbp:.2f}")
st.write(f"**SSIM** → MART: {ssim_mart:.3f}, FBP: {ssim_fbp:.3f}")

st.success("✅ Try moving the sliders → fewer angles, higher iterations. See MART stay strong where FBP collapses.")

