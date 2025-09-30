# rbyrct_core/__init__.py
import os
GPU = False
try:
    import cupy as cp  # noqa: F401
    if os.getenv("RBYRCT_FORCE_CPU") == "1":
        GPU = False
    else:
        try:
            GPU = cp.cuda.runtime.getDeviceCount() > 0
            _ = cp.zeros((1,), dtype=cp.float32)  # catches bad driver/runtime
        except Exception:
            GPU = False
except Exception:
    GPU = False

if GPU:
    from .core import forward_project, mart_reconstruct
else:
    from .cpu_core import forward_project_cpu as forward_project
    from .cpu_core import mart_reconstruct_cpu as mart_reconstruct

from .fbp_baseline import cpu_fbp_baseline
__all__ = ["forward_project", "mart_reconstruct", "cpu_fbp_baseline", "GPU"]

