import numpy as np
# from numba import njit
from collections.abc import Callable

# Type alias for any curve function
SpaceFillingCurve = Callable[[int, int], [(int, int)]]


# @njit(cache=True, fastmath=True)
def hilbert_curve(width: int, height: int) -> [(int, int)]:
    """
    Generate the Hilbert curve for the image.
    When not a power of 2, or a square an approximation is made.
    :param width: Width of the image.
    :param height: Height of the image.
    :return: List of (x, y) coordinates representing the curve.
    """

    def _rot(n, x, y, rx, ry):
        """
        Rotate/flip a quadrant appropriately.
        """
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            # Swap x and y
            x, y = y, x
        return x, y

    def _d2xy(n, d):
        """
        Convert a one-dimensional Hilbert curve index into an (x, y) coordinate.
        n is the side length (must be a power of 2) of the square.
        """
        x = 0
        y = 0
        t = d
        s = 1
        while s < n:
            rx = (t // 2) & 1
            ry = (t ^ rx) & 1
            x, y = _rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    # Determine the grid size (side length) as the next power of 2
    side = 2 ** np.ceil(np.log2(max(width, height))).astype(np.int64)
    total_points = side * side
    points = []

    # Generate Hilbert curve points in a square grid [0, side-1]
    for d in range(total_points):
        hx, hy = _d2xy(side, d)
        # Scale the Hilbert coordinate to the actual image dimensions.
        # When side == 1, we just map to 0.
        if side > 1:
            x_mapped = int(round(hx / (side - 1) * (width - 1)))
            y_mapped = int(round(hy / (side - 1) * (height - 1)))
        else:
            x_mapped, y_mapped = 0, 0
        points.append((x_mapped, y_mapped))

    # Remove duplicates while preserving order (for non-square/non-power-of-2 images)
    seen = set()
    unique_points = []
    for pt in points:
        if pt not in seen:
            seen.add(pt)
            unique_points.append(pt)
    return unique_points
