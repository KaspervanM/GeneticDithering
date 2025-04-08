from functools import partial

import pygad
import numpy as np
from main import load_image, preprocess_image, save_image, convert_chromosome_to_image, HilbertCurve

# Define your loss function (to be minimized).
# PyGAD maximizes the fitness function, so we take the negative of the loss.
def generate_fitness_function(ga_instance, solution, solution_idx):
    global originial_chromosome
    if len(original_chromosome) != len(solution):
        raise ValueError("Original and dithered images must have the same size.")
    return -sum((o - d) ** 2 for o, d in zip(original_chromosome, solution)) / len(original_chromosome)

# Parameters
population_size = 1000
num_generations = 1000


if __name__ == "__main__":
    my_image = load_image("images/groepsfoto_resized_smaller_scaled.jpg")  # Convert to grayscale
    if not my_image:
        exit()

    curve = HilbertCurve(my_image.size[0], my_image.size[1])


    def on_generation(ga):
        ga.logger.info(f"Generation = {ga.generations_completed}")
        intermediate_solution, fitness, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
        ga.logger.info(
            f"Fitness    = {fitness}")
        if intermediate_solution is not None:
            save_image(convert_chromosome_to_image(intermediate_solution, my_image.size[0], my_image.size[1], curve), "groepsfoto_resized_smaller_scaled_dithered.png")

    # Preprocess the image
    original_chromosome = preprocess_image(my_image, curve)

    chromosome_length = len(original_chromosome)

    # Dither
    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=10,
        fitness_func=generate_fitness_function,
        sol_per_pop=population_size,
        num_genes=chromosome_length,
        gene_type=int,  # Ensure 0/1 values
        init_range_low=0,
        init_range_high=2,  # Upper bound exclusive
        mutation_by_replacement=True,
        mutation_type="random",
        crossover_type="single_point",
        gene_space=[0, 1],  # Only 0s and 1s
        keep_parents=2,
        on_generation=on_generation,
        parallel_processing=['thread', 20],  # Use 20 threads for parallel processing

    )

    print("Starting dithering...")
    ga_instance.run()

    # Results
    solution, solution_fitness, _ = ga_instance.best_solution()
    print("Best solution:", solution)
    print("Best solution fitness (negative loss):", solution_fitness)
    print("Loss (you minimize):", -solution_fitness)



    save_image(convert_chromosome_to_image(solution, my_image.size[0], my_image.size[1], curve), "groepsfoto_resized_smaller_scaled_dithered.png")
