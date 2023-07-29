import numpy as np
import nibabel as nib

def dvars(img: nib.Nifti1Image) -> np.array:
    """
    Calculate DVRS metric on Nibable iamge `img`.

    DVAS measures the difference in the voxel values between to volumes.
    formula: DVARS = mean(voxel differences^2)

    Parameters
    ----------
    img: nibable image

    Returns
    -------
    dvars: 1D array
        One-dimensional array with n-1 elements, where n is number of volumes in `img`.
    """
    data = img.get_fdata()
    data_2d = data.reshape(-1, data.shape[-1])

    diff = np.diff(data_2d, axies=0)
    dvars = np.sqrt(np.mean(np.square(diff)))

    return dvars
