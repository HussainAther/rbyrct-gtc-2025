import cupy as cp
import numpy as np
from skimage.transform import iradon

def cpu_fbp_baseline(sino_cp):
    sino = cp.asnumpy(sino_cp)
    th = np.linspace(0, np.pi, sino.shape[0], endpoint=False)
    rec = iradon(sino.T, theta=th*180/np.pi, circle=False, filter_name="ramp")
    return cp.asarray(rec, dtype=cp.float32)

