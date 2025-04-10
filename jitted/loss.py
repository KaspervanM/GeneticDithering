import common

from collections.abc import Callable
import numpy as np
import numpy.typing as npt
# from numba import njit
from skimage.metrics import structural_similarity

# Type alias for any loss function
LossFunction = Callable[[common.ImageChromosome, common.Chromosome], float]


# @njit(cache=True, fastmath=True, parallel=True)
def mse(original_image: common.ImageChromosome, dithered_image: common.Chromosome) -> float:
    """
    Calculate the mean squared error loss between the original and dithered images.
    :param original_image: Original image chromosome.
    :param dithered_image: Dithered image chromosome.
    :return: Mean squared error loss value.
    """
    if original_image.shape != dithered_image.shape:
        raise ValueError(f"Original {original_image.shape} and dithered {dithered_image.shape} images must have the same size.")
    # return sum((o - d) ** 2 for o, d in zip(original_image, dithered_image)) / len(original_image)
    # return np.sum((original_image - dithered_image) ** 2, axis=-1) / original_image.shape[0]
    return ((original_image - dithered_image) ** 2).mean()


# @njit(cache=True, fastmath=True, parallel=True)
def mae(original_image: common.ImageChromosome, dithered_image: common.Chromosome) -> float:
    """
    Calculate the mean absolute error loss between the original and dithered images.
    :param original_image: Original image chromosome.
    :param dithered_image: Dithered image chromosome.
    :return: Mean absolute error loss value.
    """
    if original_image.shape != dithered_image.shape[-1:]:
        raise ValueError(f"Original and dithered images must have the same size.")
    # return sum(abs(o - d) for o, d in zip(original_image, dithered_image)) / len(original_image)
    # return np.sum(np.abs(original_image - dithered_image), axis=-1) / original_image.shape[0]
    return np.abs(original_image - dithered_image).mean()


# @njit(cache=True, fastmath=True)
def psnr(original_image: common.ImageChromosome, dithered_image: common.Chromosome) -> float:
    """
    Calculate the negative peak signal-to-noise ratio loss between the original and dithered images.
    :param original_image: Original image chromosome.
    :param dithered_image: Dithered image chromosome.
    :return: Peak signal-to-noise ratio loss value.
    """
    if original_image.shape != dithered_image.shape[-1:]:
        raise ValueError(f"Original and dithered images must have the same size.")

    return 10 * np.log10(mse(original_image, dithered_image))


class SSIM:
    def __init__(self, original_image: npt.NDArray[np.float64], curve: [(int, int)]):
        """
        Initialize the SSIM loss function.
        """
        self.width = original_image.shape[0]
        self.height = original_image.shape[1]
        self.original_image = original_image
        self.curve = curve

    def __call__(self, _, dithered_image: common.Chromosome) -> float:
        """
        Calculate the negative structural similarity index loss between the original and dithered images.
        :param _: Original image chromosome (not used).
        :param dithered_image: Dithered image chromosome.
        :return: Structural similarity index loss value.
        """

        # Convert to 2D arrays for SSIM calculation
        dithered = common.convert_chromosome_to_image(dithered_image, self.width, self.height, self.curve)
        # Calculate SSIM
        ssim_value = structural_similarity(self.original_image, dithered, data_range=1.0)

        # Return the opposite of SSIM value
        return 1 - ssim_value


class Filter:
    def __init__(self, original_image: npt.NDArray[np.float64], curve: [(int, int)], filter: Callable[[npt.NDArray, dict], npt.NDArray[np.float64]], kwargs: dict):
        """
        Initialize the filter-based loss function.
        :param original_image: Original image chromosome.
        :param curve: Curve for converting chromosome to image.
        :param filter: Filter function to apply to the original image.
        :param kwargs: Additional parameters for the filter function.
        """
        self.filter = filter
        self.width = original_image.shape[0]
        self.height = original_image.shape[1]
        self.original_filtered = filter(original_image, **kwargs)
        self.curve = curve
        self.kwargs = kwargs

    def __call__(self, _, dithered_image: common.Chromosome) -> float:
        """
        Calculate the negative Laplacian loss between the original and dithered images.
        :param _: Original image chromosome (not used).
        :param dithered_image: Dithered image chromosome.
        :return: Laplacian loss value.
        """
        # Convert to 2D arrays for Laplacian calculation
        dithered = common.convert_chromosome_to_image(dithered_image, self.width, self.height, self.curve)
        # Calculate MSE between the Laplacians
        return ((self.original_filtered - self.filter(dithered, **self.kwargs)) ** 2).mean()
