import numpy as np
from image_handling import save_image


# diagonal ramp: white in the top left corner, black in the bottom right corner.
# size 512x512
def generate_test_ramp():
    """
    Generate a 512x512 test ramp image with diagonal gradient.

    :return: numpy array (512, 512) with values from 1.0 (white at top-left) to 0.0 (black at bottom-right)
    """
    size = 512

    # Create coordinate meshgrids
    x = np.linspace(1.0, 0.0, size)  # From 1.0 to 0.0 horizontally
    y = np.linspace(1.0, 0.0, size)  # From 1.0 to 0.0 vertically

    # Create a 2D array where each value is the average of x and y coordinates
    # This creates a diagonal ramp effect
    xx, yy = np.meshgrid(x, y)
    ramp = (xx + yy) / 2.0

    return ramp

if __name__ == "__main__":
    ramp = generate_test_ramp()

    save_image(ramp, "../images/test_ramp.png")