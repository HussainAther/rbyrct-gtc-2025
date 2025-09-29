import numpy as np
from skimage.transform import iradon

def cpu_fbp_baseline(sino, theta=None, output_size=None, circle=True):
    """
    CPU FBP baseline using scikit-image.
    sino:       (n_angles, n_det) ndarray
    theta:      angles in degrees; if None -> uniform 0..180 (n_angles)
    output_size: reconstructed image size (int) or None
    """
    n_angles, _ = sino.shape
    if theta is None:
        theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    # skimage expects shape (n_det, n_angles)
    rec = iradon(sino.T, theta=theta, circle=circle, output_size=output_size)
    return rec.astype(np.float32)

