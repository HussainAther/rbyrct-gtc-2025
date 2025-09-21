# rbyrct_core/cpu_core.py
# Simple CPU versions to let the repo run on macOS (no CUDA). Slow but good enough for demo.

import numpy as np

def _angles(n_angles, dtype=np.float32):
    th = np.linspace(0, np.pi, n_angles, dtype=dtype)
    return np.cos(th), np.sin(th)

def forward_project_cpu(img, n_angles, n_det, det_spacing=1.0, spp=2.0):
    H, W = img.shape
    cos_t, sin_t = _angles(n_angles)
    sino = np.zeros((n_angles, n_det), dtype=np.float32)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    dt = 1.0 / float(spp)
    tmax = 1.5 * max(H, W)
    ts = np.arange(-tmax, tmax + 1e-6, dt, dtype=np.float32)

    for a in range(n_angles):
        c, snt = cos_t[a], sin_t[a]
        for d in range(n_det):
            u = (d - 0.5*(n_det-1)) * det_spacing
            x = cx + ts * c + u * (-snt)
            y = cy + ts * snt + u * ( c )
            x0 = np.floor(x).astype(int)
            y0 = np.floor(y).astype(int)
            mask = (x0 >= 0) & (x0+1 < W) & (y0 >= 0) & (y0+1 < H)
            if not np.any(mask): 
                continue
            x_m = x[mask]; y_m = y[mask]
            x0m = x0[mask]; y0m = y0[mask]
            dx = x_m - x0m; dy = y_m - y0m
            v00 = img[y0m, x0m]
            v01 = img[y0m, x0m+1]
            v10 = img[y0m+1, x0m]
            v11 = img[y0m+1, x0m+1]
            v0 = v00*(1-dx) + v01*dx
            v1 = v10*(1-dx) + v11*dx
            sino[a, d] = np.sum((v0*(1-dy) + v1*dy) * dt)
    return sino

def mart_reconstruct_cpu(y, H, W, iters=5, beta=1.0, det_spacing=1.0, spp=2.0, x0=None):
    n_angles, n_det = y.shape
    x = np.ones((H, W), dtype=np.float32) if x0 is None else x0.astype(np.float32).copy()
    eps = 1e-6
    for _ in range(iters):
        yhat = forward_project_cpu(x, n_angles, n_det, det_spacing, spp)
        r = y / np.maximum(yhat, eps)

        # crude ratio backprojection (no atomics needed on CPU)
        num = np.zeros_like(x)
        den = np.zeros_like(x)
        sino = r
        cos_t, sin_t = _angles(n_angles)
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        dt = 1.0 / float(spp)
        tmax = 1.5 * max(H, W)
        ts = np.arange(-tmax, tmax + 1e-6, dt, dtype=np.float32)

        for a in range(n_angles):
            c, snt = cos_t[a], sin_t[a]
            for d in range(n_det):
                u = (d - 0.5*(n_det-1)) * det_spacing
                r_ad = sino[a, d]
                x_ray = cx + ts * c + u * (-snt)
                y_ray = cy + ts * snt + u * ( c )
                x0 = np.floor(x_ray).astype(int)
                y0 = np.floor(y_ray).astype(int)
                mask = (x0 >= 0) & (x0+1 < W) & (y0 >= 0) & (y0+1 < H)
                if not np.any(mask):
                    continue
                x_m = x_ray[mask]; y_m = y_ray[mask]
                x0m = x0[mask]; y0m = y0[mask]
                dx = x_m - x0m; dy = y_m - y0m
                w00 = (1-dx)*(1-dy); w01 = dx*(1-dy); w10 = (1-dx)*dy; w11 = dx*dy
                add = r_ad * dt
                np.add.at(num, (y0m,   x0m  ), w00*add)
                np.add.at(num, (y0m,   x0m+1), w01*add)
                np.add.at(num, (y0m+1, x0m  ), w10*add)
                np.add.at(num, (y0m+1, x0m+1), w11*add)
                np.add.at(den, (y0m,   x0m  ), w00*dt)
                np.add.at(den, (y0m,   x0m+1), w01*dt)
                np.add.at(den, (y0m+1, x0m  ), w10*dt)
                np.add.at(den, (y0m+1, x0m+1), w11*dt)

        ratio = num / np.maximum(den, eps)
        x = np.maximum(x * np.power(np.maximum(ratio, eps), beta), 0.0)
    return x

