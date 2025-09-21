# rbyrct_core/core.py
# CUDA-accelerated RBYRCT CT reconstruction (MART) using CuPy RawKernels
# Contest-ready minimal implementation for NVIDIA GTC 2025 Golden Ticket

import cupy as cp

# ---------------------------------------------------------------------
# GPU kernels (CUDA C)
# ---------------------------------------------------------------------

_fwd_src = r"""
extern "C" __global__
void fwd_project(
    const float* __restrict__ img,  // [H*W]
    float* __restrict__ sino,       // [n_angles * n_det]
    const int H, const int W,
    const int n_angles, const int n_det,
    const float cx, const float cy,
    const float det_spacing,
    const float spp,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t
){
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_angles || d >= n_det) return;

    float c = cos_t[a];
    float snt = sin_t[a];

    float u = (d - 0.5f*(n_det-1)) * det_spacing;

    float max_dim = fmaxf(H, W);
    float tmax = 1.5f * max_dim;
    float dt = 1.0f / spp;

    float sum = 0.0f;
    for (float t = -tmax; t <= tmax; t += dt){
        float x = cx + t*c + u*(-snt);
        float y = cy + t*snt + u*( c );

        int x0 = (int)floorf(x);
        int y0 = (int)floorf(y);
        if (x0 >= 0 && x0+1 < W && y0 >= 0 && y0+1 < H){
            float dx = x - x0;
            float dy = y - y0;
            float v00 = img[y0*W + x0];
            float v01 = img[y0*W + (x0+1)];
            float v10 = img[(y0+1)*W + x0];
            float v11 = img[(y0+1)*W + (x0+1)];
            float v0 = v00*(1-dx) + v01*dx;
            float v1 = v10*(1-dx) + v11*dx;
            sum += (v0*(1-dy) + v1*dy) * dt;
        }
    }
    sino[a*n_det + d] = sum;
}
"""

_bp_src = r"""
extern "C" __global__
void backproject_ratio(
    const float* __restrict__ ratio, // [n_angles * n_det]
    float* __restrict__ num,         // [H*W]
    float* __restrict__ den,         // [H*W]
    const int H, const int W,
    const int n_angles, const int n_det,
    const float cx, const float cy,
    const float det_spacing,
    const float spp,
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t
){
    int a = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_angles || d >= n_det) return;

    float c = cos_t[a];
    float snt = sin_t[a];
    float u = (d - 0.5f*(n_det-1)) * det_spacing;

    float max_dim = fmaxf(H, W);
    float tmax = 1.5f * max_dim;
    float dt = 1.0f / spp;

    float r = ratio[a*n_det + d];

    for (float t = -tmax; t <= tmax; t += dt){
        float x = cx + t*c + u*(-snt);
        float y = cy + t*snt + u*( c );
        int x0 = (int)floorf(x);
        int y0 = (int)floorf(y);
        if (x0 >= 0 && x0+1 < W && y0 >= 0 && y0+1 < H){
            float dx = x - x0;
            float dy = y - y0;
            float w00 = (1-dx)*(1-dy);
            float w01 = dx*(1-dy);
            float w10 = (1-dx)*dy;
            float w11 = dx*dy;

            atomicAdd(&num[y0*W + x0], r * w00 * dt);
            atomicAdd(&num[y0*W + (x0+1)], r * w01 * dt);
            atomicAdd(&num[(y0+1)*W + x0], r * w10 * dt);
            atomicAdd(&num[(y0+1)*W + (x0+1)], r * w11 * dt);

            atomicAdd(&den[y0*W + x0], 1.0f * w00 * dt);
            atomicAdd(&den[y0*W + (x0+1)], 1.0f * w01 * dt);
            atomicAdd(&den[(y0+1)*W + x0], 1.0f * w10 * dt);
            atomicAdd(&den[(y0+1)*W + (x0+1)], 1.0f * w11 * dt);
        }
    }
}
"""

_update_src = r"""
extern "C" __global__
void mart_update(float* __restrict__ x,
                 const float* __restrict__ num,
                 const float* __restrict__ den,
                 const int N,
                 const float beta,
                 const float eps){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float n = num[i];
    float d = den[i];
    float ratio = n / fmaxf(d, eps);
    float upd = x[i] * powf(fmaxf(ratio, eps), beta);
    x[i] = fmaxf(upd, 0.0f);
}
"""

# ---------------------------------------------------------------------
# Compile kernels
# ---------------------------------------------------------------------

_fwd_kernel = cp.RawKernel(_fwd_src, "fwd_project")
_bp_kernel  = cp.RawKernel(_bp_src, "backproject_ratio")
_upd_kernel = cp.RawKernel(_update_src, "mart_update")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _angles(n_angles: int, dtype=cp.float32):
    th = cp.linspace(0, cp.pi, n_angles, dtype=dtype)
    return cp.cos(th), cp.sin(th)

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def forward_project(img, n_angles, n_det, det_spacing=1.0, spp=2.0):
    """Forward project an image (CuPy array) into a sinogram."""
    H, W = img.shape
    cos_t, sin_t = _angles(n_angles)
    sino = cp.zeros((n_angles, n_det), dtype=cp.float32)
    bx, by = 16, 16
    gx = (n_det + bx - 1)//bx
    gy = (n_angles + by - 1)//by
    _fwd_kernel((gx, gy), (bx, by),
        (img.astype(cp.float32).ravel(),
         sino.ravel(),
         H, W, n_angles, n_det,
         (W-1)/2.0, (H-1)/2.0,
         cp.float32(det_spacing),
         cp.float32(spp),
         cos_t, sin_t))
    return sino


def mart_reconstruct(y, H, W, iters=10, beta=1.0, det_spacing=1.0, spp=2.0, x0=None):
    """Reconstruct image from sinogram y using MART."""
    n_angles, n_det = y.shape
    cos_t, sin_t = _angles(n_angles)

    x = cp.ones((H, W), dtype=cp.float32) if x0 is None else x0.astype(cp.float32).copy()

    bx, by = 16, 16
    gx = (n_det + bx - 1)//bx
    gy = (n_angles + by - 1)//by

    for _ in range(iters):
        # Forward projection Px
        yhat = cp.zeros_like(y, dtype=cp.float32)
        _fwd_kernel((gx, gy), (bx, by),
            (x.ravel(), yhat.ravel(), H, W, n_angles, n_det,
             (W-1)/2.0, (H-1)/2.0, cp.float32(det_spacing), cp.float32(spp),
             cos_t, sin_t))

        eps = cp.float32(1e-6)
        r = y / cp.maximum(yhat, eps)

        num = cp.zeros((H, W), dtype=cp.float32)
        den = cp.zeros((H, W), dtype=cp.float32)

        _bp_kernel((gx, gy), (bx, by),
            (r.ravel(), num.ravel(), den.ravel(), H, W, n_angles, n_det,
             (W-1)/2.0, (H-1)/2.0, cp.float32(det_spacing), cp.float32(spp),
             cos_t, sin_t))

        N = H*W
        b = 256
        g = (N + b - 1)//b
        _upd_kernel((g,), (b,), (x.ravel(), num.ravel(), den.ravel(),
                                 N, cp.float32(beta), cp.float32(1e-6)))
    return x

