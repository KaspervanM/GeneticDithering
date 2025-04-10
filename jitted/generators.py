from jitted import common

import numpy as np
# from numba import njit, prange
from collections.abc import Callable

# Type alias for a function that generates chromosomes
ChromosomeGenerator = Callable[[int], common.Chromosome]


# @njit(cache=True, fastmath=True, parallel=True)
def random_chromosome(length: int) -> common.Chromosome:
    """
    Generate a random chromosome of given length.
    :param length: Length of the chromosome.
    :return: Random chromosome.
    """
    # Generate a random chromosome with values 0 or 1
    return np.random.randint(0, 2, length).astype(np.bool_)


class random_dither:
    def __init__(self, original_image: common.ImageChromosome):
        """
        Initialize the random dither generator.
        :param original_image: Original image chromosome.
        """
        self.original_image = original_image

    def __call__(self, length: int) -> common.Chromosome:
        """
        Generate a random dither chromosome.
        :param length: Length of the chromosome.
        :return: Random dither chromosome.
        """
        return np.where(self.original_image > np.random.rand(length), 1, 0).astype(np.bool_)

# @njit(cache=True, fastmath=True, parallel=True)
def generate_population(population_size: int, length: int, generator: ChromosomeGenerator) -> common.Chromosome:
    """
    Generate a population of chromosomes.
    :param population_size: Size of the population.
    :param length: Length of the chromosomes.
    :param generator: ChromosomeGenerator object.
    :return: List of chromosomes.
    """
    population = np.zeros((population_size, length), dtype=np.bool_)
    for i in range(population_size):
        population[i] = generator(length)
    return population
