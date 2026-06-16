import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter, convolve


class IncrementalLoss:
    def __init__(self, original_image: npt.NDArray[np.float64], initial_dither: npt.NDArray[np.float64]):
        self.original_image = original_image
        self.current_dither = initial_dither.copy()
        self.current_loss = 0.0
        self._initialize(initial_dither)

    def _initialize(self, initial_dither: npt.NDArray[np.float64]):
        raise NotImplementedError

    def update(self, row: int, col: int, new_val: float) -> float:
        """
        Update the loss based on a single pixel flip.
        Returns the NEW loss.
        """
        delta = self.calculate_delta(row, col, new_val)
        self.current_loss += delta
        self.current_dither[row, col] = new_val
        return self.current_loss

    def calculate_delta(self, row: int, col: int, new_val: float) -> float:
        """
        Calculate how much the loss would change if pixel (row, col) was changed to new_val.
        Does NOT update internal state.
        """
        raise NotImplementedError


class IncrementalMSE(IncrementalLoss):
    def _initialize(self, initial_dither: npt.NDArray[np.float64]):
        # MSE = mean((orig - dither)^2)
        # Sum of (orig - dither)^2 / N
        self.diff_sq = (self.original_image - initial_dither) ** 2
        self.current_sum_sq = np.sum(self.diff_sq)
        self.current_loss = self.current_sum_sq / self.original_image.size

    def calculate_delta(self, row: int, col: int, new_val: float) -> float:
        old_val = self.current_dither[row, col]
        if old_val == new_val:
            return 0.0

        orig_val = self.original_image[row, col]
        old_diff_sq = (orig_val - old_val) ** 2
        new_diff_sq = (orig_val - new_val) ** 2

        delta_sum_sq = new_diff_sq - old_diff_sq
        return delta_sum_sq / self.original_image.size


class IncrementalGaussianFilter(IncrementalLoss):
    def __init__(self, original_image: npt.NDArray[np.float64], initial_dither: npt.NDArray[np.float64], sigma: float):
        self.sigma = sigma
        # Pre-calculate the kernel
        radius = int(4 * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        y = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(x, y)
        self.kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        self.kernel /= np.sum(self.kernel)
        self.radius = radius

        # We compare Filter(Original) with Filter(Dither)
        self.filtered_original = original_image.copy()  # gaussian_filter(original_image, sigma=sigma)
        super().__init__(original_image, initial_dither)

    def _initialize(self, initial_dither: npt.NDArray[np.float64]):
        self.filtered_dither = gaussian_filter(initial_dither, sigma=self.sigma)
        self.diff = self.filtered_dither - self.filtered_original
        self.current_sum_sq = np.sum(self.diff ** 2)
        self.current_loss = self.current_sum_sq / self.original_image.size
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

        return (term1 + term2) / self.original_image.size

    def update(self, row: int, col: int, new_val: float) -> float:
        delta = self.calculate_delta(row, col, new_val)

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
        self.current_loss += delta
        self.current_dither[row, col] = new_val
        return self.current_loss


class IncrementalCombinedLoss:
    def __init__(self, losses: list[IncrementalLoss], weights: list[float]):
        self.losses = losses
        self.weights = np.array(weights)
        self.current_loss = sum(l.current_loss * w for l, w in zip(losses, weights))

    def calculate_delta(self, row: int, col: int, new_val: float) -> float:
        return sum(l.calculate_delta(row, col, new_val) * w for l, w in zip(self.losses, self.weights))

    def update(self, row: int, col: int, new_val: float) -> float:
        delta = self.calculate_delta(row, col, new_val)
        for l in self.losses:
            l.update(row, col, new_val)
        self.current_loss += delta
        return self.current_loss


class IncrementalLocalDensityLoss(IncrementalLoss):
    """
    Loss: MSE between local mean of binary image and original grayscale.
    Uses convolution for fast local density estimation.
    """

    def __init__(self, original_image, initial_dither, window_size=7):
        self.window_size = window_size
        self.kernel = np.ones((window_size, window_size), dtype=np.float64)
        self.kernel /= self.kernel.size
        super().__init__(original_image, initial_dither)

    def _initialize(self, initial_dither):
        self.local_mean = convolve(initial_dither, self.kernel, mode="reflect")
        self.diff = self.local_mean - self.original_image

        self.current_sum_sq = np.sum(self.diff ** 2)
        self.current_loss = self.current_sum_sq / self.original_image.size

    def calculate_delta(self, row, col, new_val):
        old_val = self.current_dither[row, col]
        if old_val == new_val:
            return 0.0

        delta = new_val - old_val
        r = self.window_size // 2

        h, w = self.original_image.shape

        r0 = max(0, row - r)
        r1 = min(h, row + r + 1)
        c0 = max(0, col - r)
        c1 = min(w, col + r + 1)

        # corresponding kernel coordinates (shifted into kernel space)
        kr0 = r0 - (row - r)
        kr1 = kr0 + (r1 - r0)

        kc0 = c0 - (col - r)
        kc1 = kc0 + (c1 - c0)

        kernel_slice = self.kernel[kr0:kr1, kc0:kc1]

        old_region = self.diff[r0:r1, c0:c1]

        # update local mean change
        # local_mean += delta * kernel
        # diff += same
        new_diff = old_region + delta * kernel_slice

        old_energy = np.sum(old_region ** 2)
        new_energy = np.sum(new_diff ** 2)

        return (new_energy - old_energy) / self.original_image.size

    def update(self, row, col, new_val):
        old_val = self.current_dither[row, col]
        delta = new_val - old_val
        if delta == 0:
            return self.current_loss

        r = self.window_size // 2

        h, w = self.original_image.shape

        r0 = max(0, row - r)
        r1 = min(h, row + r + 1)
        c0 = max(0, col - r)
        c1 = min(w, col + r + 1)

        # corresponding kernel coordinates (shifted into kernel space)
        kr0 = r0 - (row - r)
        kr1 = kr0 + (r1 - r0)

        kc0 = c0 - (col - r)
        kc1 = kc0 + (c1 - c0)

        kernel_slice = self.kernel[kr0:kr1, kc0:kc1]

        self.diff[r0:r1, c0:c1] += delta * kernel_slice

        self.current_dither[row, col] = new_val

        # recompute local loss incrementally
        self.current_loss += self.calculate_delta(row, col, new_val)
        return self.current_loss


class IncrementalTVLoss:
    def __init__(self, _, initial_dither):
        self.dither = initial_dither.copy()
        self.current_loss = self._full_tv(initial_dither)

    def _full_tv(self, x):
        return (np.sum(np.abs(np.diff(x, axis=0))) + np.sum(np.abs(np.diff(x, axis=1)))) / x.size

    def calculate_delta(self, r, c, new_val):
        old_val = self.dither[r, c]

        if old_val == new_val:
            return 0.0

        h, w = self.dither.shape

        change = 0.0

        # vertical neighbors
        if r > 0:
            change += abs((new_val - self.dither[r - 1, c])) - abs((old_val - self.dither[r - 1, c]))
        if r < h - 1:
            change += abs((new_val - self.dither[r + 1, c])) - abs((old_val - self.dither[r + 1, c]))

        # horizontal neighbors
        if c > 0:
            change += abs((new_val - self.dither[r, c - 1])) - abs((old_val - self.dither[r, c - 1]))
        if c < w - 1:
            change += abs((new_val - self.dither[r, c + 1])) - abs((old_val - self.dither[r, c + 1]))

        return change / self.dither.size

    def update(self, r, c, new_val):
        delta = self.calculate_delta(r, c, new_val)
        self.dither[r, c] = new_val
        self.current_loss += delta
        return self.current_loss
