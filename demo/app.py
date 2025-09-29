# demo/app.py
import io
import time
import numpy as np
import streamlit as st
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM

# Try CuPy; fall back cleanly to NumPy for displays/conversions
try:
    import cupy as cp
except Exception:  # macOS or no CUDA
    cp = None

from rbyrct_core import GPU, forward_project, mart_reconstruct, cpu_fbp_baseline

# -------------------------------
# Helpers
# -------------------------------
def to_np(x):
    """Convert CuPy array to NumPy if needed."""
    if cp is not None and hasattr(cp, "ndarray") and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)

@st.cache_resource
def make_phantom(N: int) -> np.ndarray:
    ph = shepp_logan_phantom().astype(np.float32)
    ph = resize(ph, (N, N), anti_aliasing=True, preserve_range=True).astype(np.float32)
    ph /= max(np.max(ph), 1e-6)
    return ph

def to_backend(arr_np):
    """Send NumPy to GPU if available; else return NumPy."""
    if GPU and cp is not None:
        return cp.asarray(arr_np)
    return arr_np

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="RBYRCT CUDA CT Demo", page_icon="🩻", layout="wide")
st.title("🩻 RBYRCT: CUDA-Accelerated MART for Low-Dose CT")

if GPU:
    st.success("✅ GPU backend active (CuPy/CUDA detected).")
else:
    st.warning("⚠️ CPU fallback (no CUDA). For real speedups, run on a Linux+NVIDIA GPU or Colab.")

st.markdown(
    """
**Goal:** enable safer breast imaging by keeping quality with **fewer projections** (lower dose).  
**Compare:** classical **FBP (CPU)** vs **MART (GPU/CPU)**. Use sliders to explore speed/quality trade-offs.
"""
)

# Sidebar controls (defaults chosen to be responsive on CPU)
st.sidebar.header("Scan Parameters")
N = st.sidebar.slider("Image size N (recon N×N)", 128, 1024, 256, step=64)
n_angles = st.sidebar.slider("Projections (angles)", 30, 720, 180, step=10)
n_det = st.sidebar.slider("Detector bins", 64, 1024, 256, step=64)
iters = st.sidebar.slider("MART iterations", 1, 50, 10)
beta = st.sidebar.slider("MART β (relaxation

