import numpy as np
from tqdm import trange
import os
from image_handling import load_image, save_image
import loss


def main():
    # Open an image
    print("Loading image...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "images", "test_ramp.png")

    my_image = load_image(image_path)
    if my_image is None:
        print(f"Failed to load image at {image_path}")
        exit()
    print("Image loaded successfully.")

    # Initialize the dithered image
    print("Initialising dithered image...")
    # seed = 42
    for seed in range(100):
        print(f"Using seed {seed} for random initialization.")
        rng = np.random.default_rng(seed)
        number_of_white_pixels = int(my_image.mean() * my_image.size)
        dithered_image = np.zeros_like(my_image, dtype=np.float64)
        dither_1d = dithered_image.ravel()
        dither_1d[:number_of_white_pixels] = 1.0
        rng.shuffle(dither_1d)

        # Initialize Incremental Loss
        print("Initializing incremental loss functions...")
        mse_loss = loss.IncrementalMSE(my_image, dithered_image)
        gauss_loss = loss.IncrementalGaussianFilter(my_image, dithered_image, sigma=.75)
        # tv_loss = loss.IncrementalTVLoss(my_image, dithered_image)

        gauss_factor = 1
        # combined_loss = loss.IncrementalCombinedLoss([gauss_loss, mse_loss], [gauss_factor, 1-gauss_factor])

        # combined_loss = loss.IncrementalCombinedLoss([loss.IncrementalGaussianFilter(my_image, dithered_image, sigma=1),
        #                                               loss.IncrementalGaussianFilter(my_image, dithered_image, sigma=2),
        #                                               loss.IncrementalGaussianFilter(my_image, dithered_image, sigma=4)],
        #                                              [4. / 7., 2. / 7., 1. / 7.])
        # combined_loss = loss.IncrementalCombinedLoss([gauss_loss, tv_loss], [0.0001,0.9999])
        combined_loss = gauss_loss

        current_loss = combined_loss.current_loss
        print(f"Initial loss: {current_loss:.6f}")

        # Simulated Annealing parameters
        epochs = 100
        accepted_total = 0

        os.makedirs("output", exist_ok=True)

        # Simulated Annealing loop
        print("Starting simulated annealing with epochs...")
        h, w = my_image.shape

        outside = trange(epochs, desc="Epochs")
        for epoch in outside:
            # Create a random ordered list of indices to flip
            flips = rng.permutation(my_image.size)
            accepted_epoch = 0

            # Use a faster range or tqdm for the inner loop if desired,
            # but for v3 performance we might want to keep it tight.
            for idx in flips:
                row, col = divmod(idx, w)

                old_val = dithered_image[row, col]
                new_val = 1.0 - old_val

                # Calculate delta loss
                delta_E = combined_loss.calculate_delta(row, col, new_val)

                # Accept if loss decreases
                if delta_E < 0:
                    current_loss = combined_loss.update(row, col, new_val)
                    dithered_image[row, col] = new_val
                    accepted_epoch += 1
                    accepted_total += 1

                    if accepted_total % 10000 == 0:
                        outside.set_postfix(accepted=accepted_total, loss=current_loss)
                        # save_image(dithered_image, f"output/best_{accepted_total}.png")

            outside.set_postfix(accepted=accepted_total, loss=current_loss)
            # save_image(dithered_image, f"output/best_{epoch}_{current_loss:.6f}_{seed}.png")

            if accepted_epoch == 0:
                print(f"No mutations accepted in epoch {epoch}, stopping early.")
                break

        print(f"Annealing finished. Final loss: {current_loss:.6f}")
        save_image(dithered_image, f"output/best_final_{current_loss:.6f}_{gauss_factor}_{seed}.png")


if __name__ == "__main__":
    main()
