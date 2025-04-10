import concurrent.futures

import skimage

from jitted import common
from jitted import loss
from jitted import crossover
from jitted import curves
from jitted import generators
from jitted import selection

import numpy as np
import numpy.typing as npt
from imageio.v3 import imread, imwrite

from jitted.generators import random_chromosome


# from numba import njit


def load_image(file_path: str) -> npt.NDArray[np.float64]:
    """
    Load an image from a file path.
    :param file_path: Path to the image file.
    :return: Loaded image as a numpy array.
    """
    return imread(file_path, mode="L").astype(np.float64) / 255.0


def save_image(image: npt.NDArray, file_path: str) -> None:
    """
    Save a PIL Image object to a file.
    :param image: numpy array representing the image.
    :param file_path: Path to save the image file.
    """
    # Save the image using PIL
    imwrite(file_path, (image * 255).astype(np.uint8), mode="L")
    # print(f"Image saved to {file_path}")


def preprocess_image(image: npt.NDArray[np.float64], curve: [(int, int)]) -> common.ImageChromosome:
    """
    Preprocess the image using the space-filling curve.
    :param image: PIL Image object.
    :param curve: List of (x, y) coordinates representing the curve.
    :return: List of pixel values between 0 and 1 in the order defined by the space-filling curve.
    """

    # Get pixel values in the order defined by the curve.
    pixel_values = np.ndarray(shape=(len(curve),), dtype=np.float64)
    for i, (x, y) in enumerate(curve):
        if x < image.shape[0] and y < image.shape[1]:
            pixel_values[i] = image[x, y]  # Get pixel value at (x, y)
        else:
            pixel_values[i] = 0.0  # Out of bounds
    return pixel_values


# @njit(cache=True, fastmath=True, parallel=True)
def mutate_chromosome(chromosome: common.Chromosome, mutation_rate: float) -> common.Chromosome:
    """
    Mutate a chromosome by flipping bits with a given mutation rate.
    :param chromosome: List of pixel values.
    :param mutation_rate: Probability of mutation for each pixel.
    :return: Mutated chromosome.
    """

    bit_flip_mask = np.random.rand(len(chromosome)) < mutation_rate
    mutated_chromosome = np.copy(chromosome)

    mutated_chromosome[bit_flip_mask] = 1 - mutated_chromosome[bit_flip_mask]  # Flip bits

    return mutated_chromosome


def dither(original_image: npt.NDArray[np.float64], iterations: int, population_size: int, mutation_rate: float,
           crossover_rate: float,
           crossover_degree: float, crossover_function: crossover.CrossOverFunction, loss_function: loss.LossFunction,
           population_selection_function: selection.PopulationSelectionFunction,
           space_filling_curve: curves.SpaceFillingCurve,
           generator: generators.ChromosomeGenerator) -> npt.NDArray[np.bool_]:
    """
    Perform dithering on the original image using a genetic algorithm.
    :param original_image: Original image.
    :param iterations: Number of iterations for the genetic algorithm.
    :param population_size: Size of the population.
    :param mutation_rate: Mutation rate for the genetic algorithm.
    :param crossover_rate: Crossover rate for the genetic algorithm.
    :param crossover_degree: Degree of crossover for the genetic algorithm.
    :param crossover_function: Crossover function to use.
    :param loss_function: Loss function to evaluate loss.
    :param population_selection_function: Population selection function to use.
    :param space_filling_curve: SpaceFillingCurve object for ordering pixels.
    :param generator: ChromosomeGenerator object for generating chromosomes.
    :return: Dithered image.
    """
    # Evan additions
    curve = space_filling_curve(original_image.shape[0], original_image.shape[1])
    # End Evan additions

    # Initialize population
    chromosome_length = len(curve)
    population = generators.generate_population(population_size, chromosome_length, generator)
    print("Initial population generated.")

    original_in_space_filling_curve = preprocess_image(original_image, curve)

    loss_increase_count = 10
    loss_most_recent = np.ndarray((loss_increase_count,), np.float64, np.array([np.inf] * loss_increase_count))

    prev_best = None

    hyperparameter_threshold = 0.0002

    # Main loop for genetic algorithm

    with concurrent.futures.ThreadPoolExecutor(50) as executor:
        for i in range(iterations):
            # Evaluate loss
            loss_scores = np.fromiter(
                executor.map(lambda chromosome: loss_function(original_in_space_filling_curve, chromosome), population),
                np.float64, count=population_size)
            # loss_scores = np.array(
            #     [loss_function(original_in_space_filling_curve, chromosome) for chromosome in population])
            best_score = loss_scores.min()
            index_of_best = loss_scores.argmin()
            curr_best_chromosome = population[index_of_best]
            # Save the best chromosome as an image
            if prev_best != best_score:
                save_image(common.convert_chromosome_to_image(curr_best_chromosome, original_image.shape[0],
                                                              original_image.shape[1],
                                                              curve),
                           f"progress/groepsfoto_resized_smaller_scaled_dithered_uniform_{i}.png")
            prev_best = best_score

            selected_population, selected_loss_scores = population_selection_function(population, loss_scores)

            # add the best loss score to the loss_most_recent array
            # first shift the array to the left
            loss_most_recent[:-1] = loss_most_recent[1:]
            # add the new best loss score to the end of the array
            loss_most_recent[-1] = best_score

            loss_most_recent_diff = np.diff(loss_most_recent)
            average_loss_decrease = np.average(-loss_most_recent_diff)

            print(
                f"{i}: Best loss score: {best_score}, mutation rate: {mutation_rate}, crossover rate: {crossover_rate}, crossover degree: {crossover_degree}, average loss decrease: {average_loss_decrease}")

            if average_loss_decrease < hyperparameter_threshold:
                print(
                    f"changing mutation rate, crossover rate and crossover degree because the average loss decrease: {average_loss_decrease} < {hyperparameter_threshold}")
                mutation_rate *= 0.95
                crossover_rate += (1 - crossover_rate) * 0.05
                crossover_degree *= 0.95
                hyperparameter_threshold *= 0.8

            preserved = selected_population[:2]
            new_population = np.ndarray(shape=(population_size, chromosome_length), dtype=np.bool_)
            new_population[:len(preserved)] = preserved
            # Generate new chromosomes
            for j in range(len(preserved), population_size):
                # Select parents using a weighted probability based on loss scores
                p1_index = np.random.choice(len(selected_population),
                                            p=selected_loss_scores / np.sum(selected_loss_scores))
                parent1 = selected_population[p1_index]
                p2_index = np.random.choice(len(selected_population),
                                            p=selected_loss_scores / np.sum(selected_loss_scores))
                # make sure parent2 is different from parent1
                while p2_index == p1_index:
                    p2_index = np.random.choice(len(selected_population),
                                                p=selected_loss_scores / np.sum(selected_loss_scores))
                parent2 = selected_population[p2_index]
                offspring1, offspring2 = crossover_function(parent1, parent2, crossover_rate, crossover_degree)
                new_population[j] = mutate_chromosome(offspring1, mutation_rate)

            population = new_population

    # Return the best chromosome as the dithered image
    best_chromosome = min(population, key=lambda c: loss_function(original_in_space_filling_curve, c))
    return common.convert_chromosome_to_image(best_chromosome, original_image.shape[0], original_image.shape[1], curve)


# Example usage:
if __name__ == '__main__':
    # Open an image (ensure you have an appropriate image file)
    my_image = load_image("../images/groepsfoto_resized_smaller_scaled.jpg")  # Convert to grayscale
    if my_image is None:
        exit()
    save_image(my_image, "groepsfoto_resized_smaller_scaled.png")  # Save the grayscale image

    # Compute SSIM of thresholded image
    curve = curves.hilbert_curve(my_image.shape[0], my_image.shape[1])
    img_str = preprocess_image(my_image, curve)
    thresholded = np.zeros_like(img_str, dtype=np.bool_)
    thresholded[img_str > 0.5] = 1
    thresholded[img_str <= 0.5] = 0

    save_image(skimage.filters.laplace(my_image), "laplace.png")
    save_image(skimage.filters.gaussian(my_image), "gaussian.png")
    save_image(skimage.filters.sobel(my_image), "sobel.png")
    save_image(common.convert_chromosome_to_image(thresholded, my_image.shape[0], my_image.shape[1], curve),
               "thresholded.png")

    # Loss function
    loss_str = "Gaussian"
    skimage.filters.gaussian(my_image)
    loss_func = loss.Filter(my_image, curve, skimage.filters.gaussian, {"sigma": 0.5})
    print(f"Threshold {loss_str} loss: {loss_func(img_str, thresholded)}")

    # Test the random dither chromosome generator
    random_dither_generator = generators.random_dither(img_str)
    test_chromosome = random_dither_generator(len(img_str))
    print(f"Random dither {loss_str} loss: {loss_func(img_str, test_chromosome)}")
    test_image = common.convert_chromosome_to_image(test_chromosome, my_image.shape[0], my_image.shape[1], curve)
    save_image(test_image, "test_random_dither.png")

    # Dither
    print("Starting dithering...")
    my_dithered_image = dither(my_image, iterations=9999999, population_size=500, mutation_rate=0.005,
                               crossover_rate=0.75,
                               crossover_degree=0.30,
                               crossover_function=crossover.uniform,
                               loss_function=loss_func,
                               population_selection_function=selection.best_half,
                               space_filling_curve=curves.hilbert_curve,
                               generator=random_chromosome)

    save_image(my_dithered_image, f"groepsfoto_resized_smaller_scaled_dithered_uniform_{loss_str}.png")
