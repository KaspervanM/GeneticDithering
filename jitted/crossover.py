import common

import numpy as np
# from numba import njit
from collections.abc import Callable

# Type alias for any crossover function
CrossOverFunction = Callable[
    [common.Chromosome, common.Chromosome, float, float], (common.Chromosome, common.Chromosome)]


# @njit(cache=True, fastmath=True)
def single_point(parent1: common.Chromosome, parent2: common.Chromosome, crossover_rate: float, _: float) -> (
        common.Chromosome, common.Chromosome):
    """
    Perform single-point crossover between two chromosomes.
    :param parent1: First parent chromosome.
    :param parent2: Second parent chromosome.
    :param crossover_rate: Probability of crossover.
    :param _: Unused parameter (for compatibility).
    :return: Two offspring chromosomes.
    """

    if np.random.rand() < crossover_rate:
        crossover_point = np.random.randint(1, len(parent1) - 1)
        offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return offspring1, offspring2
    else:
        return parent1, parent2


# @njit(cache=True, fastmath=True)
def two_point(parent1: common.Chromosome, parent2: common.Chromosome, crossover_rate: float,
              crossover_degree: float = 0.5) -> (common.Chromosome, common.Chromosome):
    """
    Perform two-point crossover between two chromosomes.
    :param parent1: First parent chromosome.
    :param parent2: Second parent chromosome.
    :param crossover_rate: Probability of crossover.
    :param crossover_degree: Degree of crossover (influences size between points).
    :return: Two offspring chromosomes.
    """

    if np.random.rand() < crossover_rate:
        crossover_point_distance = np.random.binomial(len(parent1), crossover_degree)
        crossover_point1 = np.random.randint(1, len(parent1) - crossover_point_distance)
        crossover_point2 = crossover_point1 + crossover_point_distance
        offspring1 = np.concatenate(
            (parent1[:crossover_point1], parent2[crossover_point1:crossover_point2], parent1[crossover_point2:]))
        offspring2 = np.concatenate(
            (parent2[:crossover_point1], parent1[crossover_point1:crossover_point2], parent2[crossover_point2:]))
        return offspring1, offspring2
    else:
        return parent1, parent2


# @njit(cache=True, fastmath=True, parallel=True)
def uniform(parent1: common.Chromosome, parent2: common.Chromosome, crossover_rate: float, _: float) -> (
        common.Chromosome, common.Chromosome):
    """
    Perform uniform crossover between two chromosomes.
    :param parent1: First parent chromosome.
    :param parent2: Second parent chromosome.
    :param crossover_rate: Probability of crossover.
    :param _: Unused parameter (for compatibility).
    :return: Two offspring chromosomes.
    """

    if np.random.rand() < crossover_rate:
        crossover_mask = np.random.rand(len(parent1)) < crossover_rate
        offspring1 = np.where(crossover_mask, parent1, parent2)
        offspring2 = np.where(crossover_mask, parent2, parent1)
        return offspring1, offspring2
    else:
        return parent1, parent2
