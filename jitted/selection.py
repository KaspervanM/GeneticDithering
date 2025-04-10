from jitted import common

import numpy as np
import numpy.typing as npt
# from numba import njit
from collections.abc import Callable

# Type alias for any selection function
PopulationSelectionFunction = Callable[[npt.NDArray[common.Chromosome], npt.NDArray[np.float64]], (
    npt.NDArray[common.Chromosome], npt.NDArray[np.float64])]


# @njit(cache=True, fastmath=True)
def best_half(population: npt.NDArray[common.Chromosome], loss_scores: npt.NDArray[np.float64]) -> (
        npt.NDArray[common.Chromosome], npt.NDArray[np.float64]):
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
