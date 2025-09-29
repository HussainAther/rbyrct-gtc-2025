from .fbp_baseline import cpu_fbp_baseline

GPU = False
try:
    import cupy  # noqa: F401
    GPU = True
except Exception:
    GPU = False

if GPU:
    from .core import forward_project, mart_reconstruct
else:
    from .cpu_core import forward_project_cpu as forward_project
    from .cpu_core import mart_reconstruct_cpu as mart_reconstruct

__all__ = ["forward_project", "mart_reconstruct", "cpu_fbp_baseline", "GPU"]

