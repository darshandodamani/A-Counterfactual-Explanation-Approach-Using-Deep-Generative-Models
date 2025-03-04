import os
import numpy as np
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

LATENT_VECTOR_DIR = "latent_vectors"  # Adjust path if needed
OUTPUT_FILE = os.path.join(LATENT_VECTOR_DIR, "all_latents.npy")

def combine_latent_vectors():
    """Combines all individual latent vector `.npy` files into a single NumPy array."""
    latent_files = sorted([f for f in os.listdir(LATENT_VECTOR_DIR) if f.endswith(".npy") and f.startswith("latent_")])

    if not latent_files:
        logging.error(" No latent vector files found. Ensure the directory contains `.npy` latent vectors.")
        return

    all_latents = []
    
    for file in latent_files:
        file_path = os.path.join(LATENT_VECTOR_DIR, file)
        try:
            latent_vector = np.load(file_path)  # Load individual latent vector
            all_latents.append(latent_vector)
        except Exception as e:
            logging.warning(f"Skipping {file} due to error: {e}")
            continue

    if not all_latents:
        logging.error(" No valid latent vectors loaded. Exiting.")
        return

    # Stack all latent vectors into a single array
    all_latents = np.vstack(all_latents)
    np.save(OUTPUT_FILE, all_latents)

    logging.info(f" Successfully combined {len(all_latents)} latent vectors into {OUTPUT_FILE}")

if __name__ == "__main__":
    combine_latent_vectors()
