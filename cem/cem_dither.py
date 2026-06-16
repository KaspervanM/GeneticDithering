import os
import numpy as np
import skimage.filters
from tqdm import trange
import loss
from image_handling import load_image, save_image

def cem_dither(original_image, num_samples=64, elite_fraction=0.1, alpha=0.2, iterations=100):
    """
    Perform dithering using the Cross-Entropy Method.
    """
    height, width = original_image.shape
    # Step 1: Initialize probability map
    p = original_image.copy()
    
    # Define loss function
    loss_funcs = [
        loss.Filter(original_image, skimage.filters.gaussian, {"sigma": 0.75}),
        loss.MSE(original_image),
    ]
    weights = np.array([0.5, 0.5])
    loss_func = loss.CombinedLoss(loss_funcs, weights)

    best_overall_loss = float('inf')
    best_overall_image = None

    os.makedirs("cem/output", exist_ok=True)

    t = trange(iterations, desc="CEM Dithering")
    for it in t:
        # Step 2: Generate candidate dither images
        # Sample x_i ~ Bernoulli(p_i)
        # Random values < p_i results in 1 with probability p_i
        samples = np.random.rand(num_samples, height, width) < p
        samples = samples.astype(np.float64)

        # Step 3: Evaluate objective
        losses = []
        for i in range(num_samples):
            l = loss_func(samples[i])
            losses.append(l)
        losses = np.array(losses)

        # Step 4: Select elites
        num_elites = max(1, int(num_samples * elite_fraction))
        elite_indices = np.argsort(losses)[:num_elites]
        elites = samples[elite_indices]
        
        current_best_loss = losses[elite_indices[0]]
        if current_best_loss < best_overall_loss:
            best_overall_loss = current_best_loss
            best_overall_image = elites[0].copy()

        # Step 5: Update probabilities
        elite_mean = np.mean(elites, axis=0)
        p = (1 - alpha) * p + alpha * elite_mean
        
        t.set_postfix(best_loss=best_overall_loss, current_loss=current_best_loss)
        
        if it % 10 == 0 or it == iterations - 1:
            save_image(best_overall_image, f"cem/output/best_it_{it}.png")
            save_image(p, f"cem/output/probs_it_{it}.png")

    return best_overall_image, p

if __name__ == "__main__":
    # Go up one level to find the images folder if we are inside cem/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, "images", "lena.png")
    
    if not os.path.exists(image_path):
        # Fallback to current dir
        image_path = "lena.png"
    
    print(f"Loading {image_path}...")
    img = load_image(image_path)
    if img is None:
        print("Failed to load image.")
    else:
        best_img, final_probs = cem_dither(img, num_samples=256, iterations=1000)
        save_image(best_img, "cem/output/final_dither.png")
        print("CEM Dithering complete. Results saved in cem/lena/")
