# This project will attempt to create a simple dithering algorithm using a genetic algorithm that solves some loss function.
# The chromosome will be the pixels of a greyscale image ordered in a 1D array according to a locality preserving space-filling curve.
# There will be multiple loss functions.
# The chromosomes will be exposed to mutations and crossovers
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt
from PIL import Image as ImageWrapper
from PIL.Image import Image
import math

Chromosome = npt.NDArray[np.bool_]
ImageChromosome = npt.NDArray[np.float64]


def load_image(file_path: str) -> Image:
    """
    Load an image from a file path.
    :param file_path: Path to the image file.
    :return: PIL Image object.
    """
    img = ImageWrapper.open(file_path)
    return img.convert('L')  # Convert to grayscale


def save_image(image: Image, file_path: str) -> None:
    """
    Save a PIL Image object to a file.
    :param image: PIL Image object.
    :param file_path: Path to save the image file.
    """
    image.save(file_path)
    # print(f"Image saved to {file_path}")


class SpaceFillingCurve(ABC):
    """
    Abstract base class for space-filling curves.
    """

    def __init__(self, width: int, height: int):
        """
        Initialize the space-filling curve.
        :param width: Width of the image.
        :param height: Height of the image.
        """
        self.width = width
        self.height = height
        self.curve = None

    @abstractmethod
    def generate_curve(self) -> [(int, int)]:
        """
        Generate the space-filling curve for the image.
        :return: List of (x, y) coordinates representing the curve.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class HilbertCurve(SpaceFillingCurve):
    """
    Class for generating a Hilbert space-filling curve.
    """

    def generate_curve(self) -> [(int, int)]:
        """
        Generate the Hilbert curve for the image.
        When not a power of 2, or a square an approximation is made.\
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

        if self.curve:
            return self.curve

        # Determine the grid size (side length) as the next power of 2
        side = 2 ** math.ceil(math.log2(max(self.width, self.height)))
        total_points = side * side
        points = []

        # Generate Hilbert curve points in a square grid [0, side-1]
        for d in range(total_points):
            hx, hy = _d2xy(side, d)
            # Scale the Hilbert coordinate to the actual image dimensions.
            # When side == 1, we just map to 0.
            if side > 1:
                x_mapped = int(round(hx / (side - 1) * (self.width - 1)))
                y_mapped = int(round(hy / (side - 1) * (self.height - 1)))
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
        self.curve = unique_points
        return self.curve


def preprocess_image(image: Image, curve: [(int, int)]) -> ImageChromosome:
    """
    Preprocess the image using the space-filling curve.
    :param image: PIL Image object.
    :param curve: List of (x, y) coordinates representing the curve.
    :return: List of pixel values between 0 and 1 in the order defined by the space-filling curve.
    """

    # Get pixel values in the order defined by the curve.
    pixel_values = np.ndarray(shape=(len(curve),), dtype=np.float64)
    for i, (x, y) in enumerate(curve):
        if x < image.size[0] and y < image.size[1]:
            pixel_values[i] = image.getpixel((x, y)) / 255.0  # Normalize to [0, 1]
        else:
            pixel_values[i] = 0.0  # Out of bounds
    return pixel_values


def convert_chromosome_to_image(chromosome: Chromosome, width: int, height: int, curve: [(int, int)]) -> Image:
    """
    Convert a chromosome (list of pixel values 0 and 1) to an image according to some space-filling curve.
    :param chromosome: List of pixel values.
    :param width: Width of the image.
    :param height: Height of the image.
    :param curve: List of (x, y) coordinates representing the curve.
    :return: PIL Image object.
    """
    if len(chromosome) != width * height:
        raise ValueError("Chromosome length must match image size.")

    # Create a new image
    new_image = ImageWrapper.new('L', (width, height))
    # Set pixel values according to the chromosome and the curve
    for i, (x, y) in enumerate(curve):
        if i < len(chromosome):
            new_image.putpixel((x, y), 255 * int(chromosome[i]))  # Assuming binary values 0 or 1
    return new_image


def mutate_chromosome(chromosome: Chromosome, mutation_rate: float,
                      random_generator: np.random.RandomState = np.random.RandomState()) -> Chromosome:
    """
    Mutate a chromosome by flipping bits with a given mutation rate.
    :param chromosome: List of pixel values.
    :param mutation_rate: Probability of mutation for each pixel.
    :param random_generator: Optional random generator for reproducibility.
    :return: Mutated chromosome.
    """

    bit_flip_mask = random_generator.rand(len(chromosome)) < mutation_rate
    mutated_chromosome = np.copy(chromosome)

    mutated_chromosome[bit_flip_mask] = 1 - mutated_chromosome[bit_flip_mask]  # Flip bits

    return mutated_chromosome


class CrossOverFunction(ABC):
    """
    Abstract base class for crossover functions.
    """

    @abstractmethod
    def crossover(self, parent1: Chromosome, parent2: Chromosome, crossover_rate: float,
                  crossover_degree: Optional[float],
                  random_generator: np.random.RandomState = np.random.RandomState()) -> (
            Chromosome, Chromosome):
        """
        Perform crossover between two chromosomes.
        :param parent1: First parent chromosome.
        :param parent2: Second parent chromosome.
        :param crossover_rate: Probability of crossover.
        :param crossover_degree: Degree of crossover (useful to some implementations).
        :param random_generator: Optional random generator for reproducibility.
        :return: Two offspring chromosomes.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class SinglePointCrossover(CrossOverFunction):
    """
    Class for single-point crossover.
    """

    def crossover(self, parent1: Chromosome, parent2: Chromosome, crossover_rate: float,
                  crossover_degree: Optional[float],
                  random_generator: np.random.RandomState = np.random.RandomState()) -> (
            Chromosome, Chromosome):
        """
        Perform single-point crossover between two chromosomes.
        :param parent1: First parent chromosome.
        :param parent2: Second parent chromosome.
        :param crossover_rate: Probability of crossover.
        :param crossover_degree: Degree of crossover (not used).
        :param random_generator: Optional random generator for reproducibility.
        :return: Two offspring chromosomes.
        """

        if random_generator.rand() < crossover_rate:
            crossover_point = random_generator.randint(1, len(parent1) - 1)
            offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return offspring1, offspring2
        else:
            return parent1, parent2


class TwoPointCrossover(CrossOverFunction):
    """
    Class for two-point crossover.
    """

    def crossover(self, parent1: Chromosome, parent2: Chromosome, crossover_rate: float,
                  crossover_degree: Optional[float],
                  random_generator: np.random.RandomState = np.random.RandomState()) -> (
            Chromosome, Chromosome):
        """
        Perform two-point crossover between two chromosomes.
        :param parent1: First parent chromosome.
        :param parent2: Second parent chromosome.
        :param crossover_rate: Probability of crossover.
        :param crossover_degree: Degree of crossover (influences size between points).
        :param random_generator: Optional random generator for reproducibility.
        :return: Two offspring chromosomes.
        """

        if crossover_degree is None:
            crossover_degree = 0.5

        if random_generator.rand() < crossover_rate:
            crossover_point_distance = random_generator.binomial(len(parent1), crossover_degree)
            crossover_point1 = random_generator.randint(1, len(parent1) - crossover_point_distance)
            crossover_point2 = crossover_point1 + crossover_point_distance
            offspring1 = np.concatenate(
                (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
            offspring2 = np.concatenate(
                (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
            return offspring1, offspring2
        else:
            return parent1, parent2


class UniformCrossover(CrossOverFunction):
    """
    Class for uniform crossover.
    """

    def crossover(self, parent1: Chromosome, parent2: Chromosome, crossover_rate: float,
                  crossover_degree: Optional[float],
                  random_generator: np.random.RandomState = np.random.RandomState()) -> (
            Chromosome, Chromosome):
        """
        Perform uniform crossover between two chromosomes.
        :param parent1: First parent chromosome.
        :param parent2: Second parent chromosome.
        :param crossover_rate: Probability of crossover.
        :param crossover_degree: Degree of crossover (the probability for each pixel to be included in the crossover).
        :param random_generator: Optional random generator for reproducibility.
        :return: Two offspring chromosomes.
        """

        if random_generator.rand() < crossover_rate:
            crossover_mask = random_generator.rand(len(parent1)) < crossover_rate
            offspring1 = np.where(crossover_mask, parent1, parent2)
            offspring2 = np.where(crossover_mask, parent2, parent1)
            return offspring1, offspring2
        else:
            return parent1, parent2


class ChromosomeGenerator(ABC):
    """
    Abstract base class for generating chromosomes.
    """

    def __init__(self, length: int, random_generator: np.random.RandomState = np.random.RandomState()):
        """
        Initialize the chromosome generator.
        :param length: Length of the chromosome.
        :param random_generator: Optional random generator for reproducibility.
        """
        self.length = length
        self._random_generator = random_generator

    @abstractmethod
    def generate_chromosome(self) -> Chromosome:
        """
        Generate a random chromosome of given length.
        :return: Random chromosome.
        """
        pass


class RandomChromosomeGenerator(ChromosomeGenerator):
    """
    Class for generating random chromosomes.
    """

    def generate_chromosome(self) -> Chromosome:
        """
        Generate a random chromosome of given length.
        :return: Random chromosome.
        """
        # Generate a random chromosome with values 0 or 1
        return self._random_generator.randint(0, 2, self.length).astype(np.bool_)


def generate_population(population_size: int, generator: ChromosomeGenerator) -> Chromosome:
    """
    Generate a population of chromosomes.
    :param population_size: Size of the population.
    :param generator: ChromosomeGenerator object.
    :return: List of chromosomes.
    """
    population = np.ndarray((population_size, generator.length), dtype=np.bool_)
    for i in range(population_size):
        population[i] = generator.generate_chromosome()
    return population


class LossFunction(ABC):
    """
    Abstract base class for loss functions.
    """

    @abstractmethod
    def calculate_loss(self, original_image_chromosome: ImageChromosome,
                       dithered_image_chromosome: Chromosome) -> npt.NDArray[np.float64]:
        """
        Calculate the loss between the original and dithered images.
        :param original_image_chromosome: Chromosome representing the original image (Not limited to only 0 and 1, also inbetween).
        :param dithered_image_chromosome: Chromosome representing the dithered image.
        :return: Loss value.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class MeanSquaredErrorLoss(LossFunction):
    """
    Class for calculating the mean squared error loss.
    """
    def calculate_loss(self, original_image: ImageChromosome, dithered_image: Chromosome) -> npt.NDArray[np.float64]:
        """
        Calculate the mean squared error loss between the original and dithered images.
        :param original_image: Original image chromosome.
        :param dithered_image: Dithered image chromosome.
        :return: Mean squared error loss value.
        """
        if original_image.shape != dithered_image.shape[-1:]:
            raise ValueError(f"Original and dithered images must have the same size.")
        # return sum((o - d) ** 2 for o, d in zip(original_image, dithered_image)) / len(original_image)
        # return np.sum((original_image - dithered_image) ** 2, axis=-1) / original_image.shape[0]
        return ((original_image - dithered_image) ** 2).mean(axis=-1)

class MeanAbsoluteErrorLoss(LossFunction):
    """
    Class for calculating the mean absolute error loss.
    """

    def calculate_loss(self, original_image: ImageChromosome, dithered_image: Chromosome) -> npt.NDArray[np.float64]:
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
        return np.abs(original_image - dithered_image).mean(axis=-1)

class PeakSignalToNoiseRatioLoss(LossFunction):
    """
    Class for calculating the peak signal-to-noise ratio (PSNR) loss.
    """

    def calculate_loss(self, original_image: ImageChromosome, dithered_image: Chromosome) -> npt.NDArray[np.float64]:
        """
        Calculate the peak signal-to-noise ratio loss between the original and dithered images.
        :param original_image: Original image chromosome.
        :param dithered_image: Dithered image chromosome.
        :return: Peak signal-to-noise ratio loss value.
        """
        if original_image.shape != dithered_image.shape[-1:]:
            raise ValueError(f"Original and dithered images must have the same size.")

        mse = MeanSquaredErrorLoss().calculate_loss(original_image, dithered_image)
        # if mse == 0:
        #     return float('-inf')  # No noise
        # return -20 * math.log10(1 / math.sqrt(mse))
        return -10 * np.log10(mse)


class PopulationSelectionFunction(ABC):
    """
    Abstract base class for population selection functions.
    """

    @abstractmethod
    def select(self, population: npt.NDArray[Chromosome], loss_scores: npt.NDArray[np.float64]) -> (
            npt.NDArray[Chromosome], npt.NDArray[np.float64]):
        """
        Select a new population based on loss scores.
        :param population: Current population of chromosomes.
        :param loss_scores: Loss scores of the current population.
        :return: Selected new population in order of importance and the corresponding loss scores.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class SelectBestHalf(PopulationSelectionFunction):
    """
    Class for selecting the best half of the population.
    """

    def select(self, population: npt.NDArray[Chromosome], loss_scores: npt.NDArray[np.float64]) -> (
            npt.NDArray[Chromosome], npt.NDArray[np.float64]):
        """
        Select the best half of the population based on loss scores.
        :param population: Current population of chromosomes.
        :param loss_scores: Loss scores of the current population.
        :return: Selected new population in order of importance and the corresponding loss scores.
        """
        sorted_indices = np.argsort(loss_scores)
        half_size = len(population) // 2
        selected_population = population[sorted_indices[:half_size]]
        selected_loss_scores = loss_scores[sorted_indices[:half_size]]
        return selected_population, selected_loss_scores



def dither(original_image: Image, iterations: int, population_size: int, mutation_rate: float, crossover_rate: float,
           crossover_degree: float, crossover_function: CrossOverFunction, loss_function: LossFunction,
           population_selection_function: PopulationSelectionFunction, space_filling_curve: SpaceFillingCurve) -> Image:
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
    :return: Dithered image.
    """
    # Evan additions
    curve = space_filling_curve.generate_curve()
    rng = np.random.RandomState(42)  # For reproducibility and also not creating a new one every time (that's bad)
    # End Evan additions

    # Initialize population
    chromosome_length = len(curve)
    generator = RandomChromosomeGenerator(chromosome_length, rng)
    population = generate_population(population_size, generator)
    print("Initial population generated.")

    original_in_space_filling_curve = preprocess_image(original_image, curve)

    loss_increase_count = 10
    loss_most_recent = np.ndarray((loss_increase_count,), np.float64, np.array([np.inf] * loss_increase_count))

    prev_best = None

    hyperparameter_threshold = 0.0002

    # Main loop for genetic algorithm
    for i in range(iterations):
        # Evaluate loss
        # loss_scores = [loss_function.calculate_loss(original_in_space_filling_curve, chromosome) for chromosome in population]
        # with concurrent.futures.ThreadPoolExecutor(20) as executor:
        #     loss_scores = np.fromiter(executor.map(
        #         lambda chromosome: loss_function.calculate_loss(original_in_space_filling_curve, chromosome),
        #         population
        #     ), np.float64, count=population_size)
        # loss_scores = np.fromiter(
        #     (loss_function.calculate_loss(original_in_space_filling_curve, chromosome) for chromosome in population),
        #     np.float64)
        loss_scores = np.array(
            [loss_function.calculate_loss(original_in_space_filling_curve, chromosome) for chromosome in population])
        # loss_scores = loss_function.calculate_loss(original_in_space_filling_curve, population)
        best_score = loss_scores.min()
        index_of_best = loss_scores.argmin()
        curr_best_chromosome = population[index_of_best]
        # Save the best chromosome as an image
        if prev_best != best_score:
            save_image(convert_chromosome_to_image(curr_best_chromosome, original_image.size[0], original_image.size[1],
                                                   curve),
                       f"progress/groepsfoto_resized_smaller_scaled_dithered_uniform_{i}.png")
        prev_best = best_score

        selected_population, selected_loss_scores = population_selection_function.select(population, loss_scores)

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
            mutation_rate *= 0.75
            crossover_rate += (1 - crossover_rate) * 0.1
            crossover_degree *= 0.9
            hyperparameter_threshold *= 0.8

        # print(selected_loss_scores[:10])
        preserved = selected_population[:2]
        new_population = np.ndarray(shape=(population_size, chromosome_length), dtype=np.bool_)
        new_population[:len(preserved)] = preserved
        # Generate new chromosomes
        for j in range(len(preserved), population_size):
            # Select parents using a weighted probability based on loss scores
            p1_index = rng.choice(len(selected_population), p=selected_loss_scores / np.sum(selected_loss_scores))
            parent1 = selected_population[p1_index]
            p2_index = rng.choice(len(selected_population), p=selected_loss_scores / np.sum(selected_loss_scores))
            # make sure parent2 is different from parent1
            while p2_index == p1_index:
                p2_index = rng.choice(len(selected_population),
                                      p=selected_loss_scores / np.sum(selected_loss_scores))
            parent2 = selected_population[p2_index]
            offspring1, offspring2 = crossover_function.crossover(parent1, parent2, crossover_rate, crossover_degree,
                                                                  rng)
            new_population[j] = mutate_chromosome(offspring1, mutation_rate, rng)

        population = new_population

    # Return the best chromosome as the dithered image
    # best_chromosome = min(population, key=lambda c: loss_function.calculate_loss(original_in_space_filling_curve, c))
    best_chromosome = population[loss_function.calculate_loss(original_in_space_filling_curve, population).argmin()]
    return convert_chromosome_to_image(best_chromosome, original_image.size[0], original_image.size[1], curve)


# Example usage:
if __name__ == '__main__':
    # Open an image (ensure you have an appropriate image file)
    my_image = load_image("images/groepsfoto_resized_smaller_scaled.jpg")  # Convert to grayscale
    if not my_image:
        exit()

    # Dither
    print("Starting dithering...")
    my_dithered_image = dither(my_image, iterations=6000, population_size=1000, mutation_rate=0.005,
                               crossover_rate=0.75,
                               crossover_degree=0.30,
                               crossover_function=UniformCrossover(),
                               loss_function=MeanSquaredErrorLoss(),
                               population_selection_function=SelectBestHalf(),
                               space_filling_curve=HilbertCurve(my_image.size[0], my_image.size[1]))

    save_image(my_dithered_image, "groepsfoto_resized_smaller_scaled_dithered_uniform.png")
