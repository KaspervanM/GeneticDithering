from collections.abc import Callable
import warnings
import numpy as np
import numpy.typing as npt
from skimage.metrics import structural_similarity

# Type alias for any loss function
LossFunction = Callable[[npt.NDArray[np.float64]], float]


# Loss functions
class Loss:
    def __init__(self, original_image: npt.NDArray[np.float64]):
        """
        Initialize the loss function.
        :param original_image: Original image.
        """
        self.original_image = original_image
        self.out = np.zeros_like(original_image, dtype=np.float64)

    def __call__(self, dithered_image: npt.NDArray[np.float64]) -> float:
        """
        Calculate the loss value.
        :param dithered_image: Dithered image.
        :return: Loss value.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __repr__(self):
        """
        String representation of the loss function.
        :return: String representation.
        """
        return f"{self.__class__.__name__}()"


class MSE(Loss):
    def __call__(self, dithered_image: npt.NDArray[np.float64]) -> float:
        """
        Calculate the Mean Squared Error (MSE) loss.
        :param dithered_image: Dithered image.
        :return: MSE loss value.
        """
        # Do it in place to save memory
        np.subtract(self.original_image, dithered_image, out=self.out)
        np.square(self.out, out=self.out)
        return self.out.mean()


class SSIM(Loss):
    def __init__(self, original_image: npt.NDArray[np.float64], kwargs: dict = None):
        """
        Initialize the SSIM loss function.
        :param original_image: Original image.
        :param kwargs: Additional parameters for SSIM calculation.
        """
        super().__init__(original_image)
        self.kwargs = kwargs if kwargs is not None else {}

    def __call__(self, dithered_image: npt.NDArray[np.float64]) -> float:
        """
        Calculate the SSIM loss between the original and dithered images.
        :param dithered_image: Dithered image.
        :return: SSIM loss value.
        """
        return 1 - structural_similarity(self.original_image, dithered_image, data_range=1.0, **self.kwargs)

    def __repr__(self):
        """
        String representation of the SSIM loss function.
        :return: String representation.
        """
        return f"{self.__class__.__name__}({self.kwargs})"


class Filter(Loss):
    def __init__(self, original_image: npt.NDArray[np.float64],
                 filter: Callable[[npt.NDArray, dict], npt.NDArray],
                 kwargs: dict = None):
        """
        Initialize the filter-based loss function.
        :param original_image: Original image.
        :param filter: Filter function to apply to the original image.
        :param kwargs: Additional parameters for the filter function.
        """
        super().__init__(filter(original_image, **kwargs))
        self.filter = filter
        self.kwargs = kwargs

    def __call__(self, dithered_image: npt.NDArray[np.float64]) -> float:
        """
        Calculate the filtered MSE loss between the original and dithered images.
        :param dithered_image: Dithered image.
        :return: Loss value.
        """
        self.out = self.filter(dithered_image, **self.kwargs)
        np.subtract(self.original_image, self.out, out=self.out)
        np.square(self.out, out=self.out)
        return self.out.mean()

    def __repr__(self):
        """
        String representation of the Filter loss function.
        :return: String representation.
        """
        kwargs_copy = self.kwargs.copy()
        return f"{self.__class__.__name__}.{self.filter.__name__}({kwargs_copy})"


class CombinedLoss:
    def __init__(self, loss_functions: [LossFunction], weights: npt.NDArray[np.float64]):
        """
        Initialize the combined loss function.
        :param loss_functions: List of loss functions.
        :param weights: Weights for each loss function.
        """
        self.loss_functions = loss_functions
        self.weights = weights
        if len(loss_functions) != weights.shape[0]:
            raise ValueError("Number of loss functions must match number of weights.")
        if sum(weights) != 1:
            warnings.warn("Sum of weights is not 1. Are you sure about this?")
        if any(w < 0 for w in weights):
            warnings.warn("Some weights are negative. Are you sure about this?")
        if any(not callable(loss) for loss in loss_functions):
            raise ValueError("All loss functions must be callable.")
        self.out = np.zeros(len(loss_functions), dtype=np.float64)

    def __call__(self, dithered_image: npt.NDArray[np.float64]) -> float:
        """
        Calculate the combined loss value.
        :param dithered_image: Dithered image.
        :return: Combined loss value.
        """
        for i, loss in enumerate(self.loss_functions):
            self.out[i] = loss(dithered_image)
        return np.dot(self.out, self.weights)

    def __repr__(self):
        """
        String representation of the combined loss function.
        :return: String representation.
        """
        loss_descriptions = [f"{loss} * {weight:.2f}" for loss, weight in zip(self.loss_functions, self.weights)]
        return f"{self.__class__.__name__}(\n\t" + "\n\t".join(loss_descriptions) + "\n)"
