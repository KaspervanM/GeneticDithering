import numpy as np
from imageio.v3 import imread, imwrite
import numpy.typing as npt

def load_image(file_path: str) -> npt.NDArray[np.float64]:
    """
    Load an image from a file path.
    :param file_path: Path to the image file.
    :return: Loaded image as a numpy array.
    """
    return imread(file_path, mode="L").astype(np.float64) / 255.0


def save_image(image: npt.NDArray, file_path: str) -> None:
    """
    Save an image to a file path.
    :param image: numpy array representing the image.
    :param file_path: Path to save the image file.
    """
    imwrite(file_path, (image * 255).astype(np.uint8), mode="L")
