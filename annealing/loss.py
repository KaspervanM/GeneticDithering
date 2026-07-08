import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve


class IncrementalLoss:
    def __init__(self, original_image: npt.NDArray[np.float64], initial_dither: npt.NDArray[np.float64]):
        self.original_image = original_image
        self.current_dither = initial_dither.copy()
        self.current_loss = 0.0
        self.delta = 0.0

    def calculate_delta(self, row: int, col: int, new_val: float) -> float:
        """
        Calculate how much the loss would change if pixel (row, col) was changed to new_val.
        Does NOT update internal state.
        NB: call before `update`.
        """
        raise NotImplementedError

    def update(self, row: int, col: int, new_val: float) -> float:
        """
        Update the loss based on a single pixel flip.
        Returns the NEW loss.
        NB: call `calculate_delta` first.
        """
        raise NotImplementedError


class IncrementalMSE(IncrementalLoss):
    def __init__(self, original_image: npt.NDArray[np.float64], initial_dither: npt.NDArray[np.float64]):
        super().__init__(original_image, initial_dither)
        # MSE = mean((orig - dither)^2)
        # Sum of (orig - dither)^2 / N
        self.diff_sq = (original_image - initial_dither) ** 2
        self.current_sum_sq = np.sum(self.diff_sq)
        self.current_loss = self.current_sum_sq / original_image.size

    def calculate_delta(self, row: int, col: int, new_val: float) -> float:
        old_val = self.current_dither[row, col]
        if old_val == new_val:
            return 0.0

        orig_val = self.original_image[row, col]
        old_diff_sq = (orig_val - old_val) ** 2
        new_diff_sq = (orig_val - new_val) ** 2

        delta_sum_sq = new_diff_sq - old_diff_sq
        self.delta = delta_sum_sq / self.original_image.size
        return self.delta

    def update(self, row: int, col: int, new_val: float) -> float:
        self.current_loss += self.delta
        self.current_dither[row, col] = new_val
        return self.current_loss


class IncrementalGaussianFilter(IncrementalLoss):
    def __init__(self, original_image: npt.NDArray[np.float64], initial_dither: npt.NDArray[np.float64], sigma: float):
        super().__init__(original_image, initial_dither)
        self.sigma = sigma

        # Pre-calculate the kernel
        radius = int(4 * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        y = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(x, y)
        self.kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        self.kernel /= np.sum(self.kernel)
        self.radius = radius

        # We compare Original with Gaussian(Dither)
        self.diff = convolve(initial_dither, self.kernel, mode="reflect") - original_image
        self.current_sum_sq = np.sum(self.diff ** 2)
        self.current_loss = self.current_sum_sq / original_image.size

    # TODO: fix edges. currently wrongly clipping while we should be reflecting.
    def calculate_delta(self, row: int, col: int, new_val: float) -> float:
        old_val = self.current_dither[row, col]
        val_diff = new_val - old_val
        if val_diff == 0:
            return 0.0

        h, w = self.original_image.shape
        r = self.radius

        # Region affected by the flip
        r_start, r_end = max(0, row - r), min(h, row + r + 1)
        c_start, c_end = max(0, col - r), min(w, col + r + 1)

        # Kernel slice
        kr_start, kr_end = r - (row - r_start), r + (r_end - row)
        kc_start, kc_end = r - (col - c_start), r + (c_end - col)

        kernel_slice = self.kernel[kr_start:kr_end, kc_start:kc_end]

        # Current diff in the affected region
        diff_slice = self.diff[r_start:r_end, c_start:c_end]

        # New diff = diff + val_diff * kernel
        # New MSE sum = sum((diff + val_diff * kernel)^2)
        #            = sum(diff^2 + 2 * diff * val_diff * kernel + (val_diff * kernel)^2)
        # Delta sum  = sum(2 * diff * val_diff * kernel + (val_diff * kernel)^2)

        term1 = 2 * val_diff * np.sum(diff_slice * kernel_slice)
        term2 = (val_diff ** 2) * np.sum(kernel_slice ** 2)

        self.delta = (term1 + term2) / self.original_image.size
        return self.delta

    def update(self, row: int, col: int, new_val: float) -> float:
        old_val = self.current_dither[row, col]
        val_diff = new_val - old_val

        h, w = self.original_image.shape
        r = self.radius
        r_start, r_end = max(0, row - r), min(h, row + r + 1)
        c_start, c_end = max(0, col - r), min(w, col + r + 1)
        kr_start, kr_end = r - (row - r_start), r + (r_end - row)
        kc_start, kc_end = r - (col - c_start), r + (c_end - col)

        kernel_slice = self.kernel[kr_start:kr_end, kc_start:kc_end]

        # Update internal filtered image state
        self.diff[r_start:r_end, c_start:c_end] += val_diff * kernel_slice
        self.current_loss += self.delta
        self.current_dither[row, col] = new_val
        return self.current_loss


class IncrementalCombinedLoss:
    def __init__(self, losses: list[IncrementalLoss], weights: list[float]):
        self.losses = losses
        self.weights = np.array(weights)
        self.current_loss = sum(l.current_loss * w for l, w in zip(losses, weights))
        self.delta = 0.0

    def calculate_delta(self, row: int, col: int, new_val: float) -> float:
        self.delta = sum(l.calculate_delta(row, col, new_val) * w for l, w in zip(self.losses, self.weights))
        return self.delta

    def update(self, row: int, col: int, new_val: float) -> float:
        for l in self.losses:
            l.update(row, col, new_val)
        self.current_loss += self.delta
        return self.current_loss
