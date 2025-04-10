import numpy as np
import numpy.typing as npt

Chromosome = npt.NDArray[np.bool_]
ImageChromosome = npt.NDArray[np.float64]


def convert_chromosome_to_image(chromosome: Chromosome, width: int, height: int,
                                curve: [(int, int)]) -> npt.NDArray[np.bool_]:
    """
    Convert a chromosome (list of pixel values 0 and 1) to an image according to some space-filling curve.
    :param chromosome: List of pixel values.
    :param width: Width of the image.
    :param height: Height of the image.
    :param curve: List of (x, y) coordinates representing the curve.
    :return: Converted image as a numpy array.
    """
    if len(chromosome) != width * height:
        raise ValueError(f"Chromosome length ({len(chromosome)}) does not match image size ({width * height}).")

    # Create a new image
    new_image = np.zeros((width, height), dtype=np.bool_)
    # Set pixel values according to the chromosome and the curve
    for i, (x, y) in enumerate(curve):
        if i < len(chromosome):
            new_image[x, y] = chromosome[i]  # Get pixel value at (x, y)
    return new_image
