# 🩻 CUDA-Accelerated RBYRCT (MART) for Low-Dose CT

Open-source GPU reconstruction using MART with custom CuPy RawKernels.  
Built for the **NVIDIA GTC 2025 Golden Ticket Contest**.

- ⚡ **XX× faster** than CPU FBP on 512×512 CT
- 📈 **Higher PSNR/SSIM** than FBP at **YY% fewer projections**
- 🎛️ **Interactive demo:** real-time sliders for angles, detectors, iterations
- 🧪 **Notebook benchmarks** + reproducible Docker setup
- 🎯 **Goal:** Enable *lower-dose breast CT* → less radiation, higher diagnostic confidence

---

## ✨ Why this project?

- **GPU unlocks MART:** Iterative MART recon is too slow on CPU, but fast + interactive on GPU.  
- **Dose reduction story:** Better recon from fewer projections → lower patient exposure.  
- **Distinctive:** Most entries will showcase LLMs. This is **CT reconstruction for healthcare**.

---

## 📦 Installation

### Prerequisites
- NVIDIA GPU (for GPU path) + CUDA 12.x  
- Python 3.10–3.11

### Option 1 — pip
```bash
git clone https://github.com/<your-handle>/rbyrct-gtc-2025.git
cd rbyrct-gtc-2025
pip install -r requirements.txt
pip install -e .
````

### Option 2 — Conda

```bash
conda create -n rbyrct python=3.11
conda activate rbyrct
pip install -r requirements.txt
pip install -e .
```

### Option 3 — Docker (NGC-ready)

GPU:

```bash
docker build -f Dockerfile.gpu -t rbyrct:gpu .
docker run --gpus all -p 8501:8501 rbyrct:gpu
```

CPU:

```bash
docker build -f Dockerfile.cpu -t rbyrct:cpu .
docker run -p 8501:8501 rbyrct:cpu
```

---

## 🚀 Quickstart

### Launch the demo (Streamlit app)

```bash
streamlit run demo/app.py
```

* Adjust angles, detectors, and iterations with sliders
* Compare **CPU-FBP vs GPU-MART** visually + numerically

### Run benchmark (notebook or script)

```bash
python examples/bench.py
```

Or open [`notebooks/colab_demo.ipynb`](notebooks/colab_demo.ipynb) in Google Colab.

---

## 📊 Results

Benchmarked on `GPU_NAME` vs `CPU_NAME` (fill with your hardware):

| Setting                    | CPU FBP (ms) | GPU MART (ms) | Speedup |
| -------------------------- | ------------ | ------------- | ------- |
| 512×512, 180 proj, det=512 | AAA          | BBB           | CCC×    |

Quality comparison (PSNR/SSIM):

* 180 → 120 projections:

  * FBP = **XX dB**, SSIM = **0.YYY**
  * MART = **ZZ dB**, SSIM = **0.WWW**

---

## 🖼️ Screenshots

| Nsight Timeline        | Side-by-side Recon         |
| ---------------------- | -------------------------- |
| ![](assets/nsight.png) | ![](assets/comparison.png) |

---

## 🧰 Tech stack

* **CUDA via CuPy RawKernels**
* **Nsight Systems/Compute profiling**
* **Streamlit interactive demo**
* **scikit-image FBP baseline**
* **Docker (CPU/GPU) for reproducibility**
* **Hugging Face Spaces / Streamlit Cloud deploy**

---

## 📂 Repo layout

```
rbyrct-gtc-2025/
├─ rbyrct_core/        # CuPy kernels + MART loop
├─ demo/               # Streamlit UI (interactive app)
├─ examples/           # Benchmarks + notebooks
├─ assets/             # Screenshots, figures
├─ tests/              # Simple smoke tests
├─ Dockerfile.gpu      # GPU-ready Docker build
├─ Dockerfile.cpu      # CPU-only Docker build
└─ README.md
```

---

## ⚡ API usage

```python
import cupy as cp
from rbyrct_core import forward_project, mart_reconstruct, cpu_fbp_baseline

x = cp.ones((512, 512), dtype=cp.float32)

# Forward projection
sino = forward_project(x, n_angles=180, n_det=512)

# MART reconstruction (GPU)
rec = mart_reconstruct(sino, H=512, W=512, iters=10, beta=1.0)

# CPU baseline
rec_fbp = cpu_fbp_baseline(cp.asnumpy(sino))
```

---

## 📝 Citation

If you use this code, please cite:

```
CUDA-Accelerated RBYRCT (MART) for Low-Dose CT
NVIDIA GTC 2025 Golden Ticket Contest
https://github.com/<your-handle>/rbyrct-gtc-2025
```

See `CITATION.cff` for metadata.

---

## 📣 Social Post (ready-to-share)

> Open-sourced a CUDA-accelerated RBYRCT pipeline for low-dose CT.
> MART + custom kernels give **XX× speedup** over CPU and **better PSNR/SSIM** than FBP at **YY% fewer projections**.
> Notebook + demo inside. Built on CUDA/cuFFT + profiled with Nsight.
> #NVIDIAGTC @NVIDIAGTC @NVIDIADeveloper

---

## ⚠️ Disclaimer

This project is for **educational/demo purposes only**.
It is **not a clinical CT reconstruction tool**.
Use in healthcare settings requires regulatory approval.

