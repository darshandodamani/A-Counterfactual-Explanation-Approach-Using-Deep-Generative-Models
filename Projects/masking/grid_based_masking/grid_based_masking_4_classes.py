import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import logging

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------------------
# Add Python Paths for local modules
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "autoencoder")))
from encoder import VariationalEncoder
from decoder import Decoder
from classifiers_4_class import ClassifierModel as Classifier

sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
from initialize_masking_pipeline_4_class import INITIAL_PREDICTIONS_CSV as initial_predictions_csv

# ------------------------------------------------------------------------------
# Constants and Paths
# ------------------------------------------------------------------------------
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_4_classes.pth"
OUTPUT_CSV = "results/masking/grid_based_masking_4_classes_results.csv"
TEST_DIR = "dataset/town7_dataset/test/"
DEBUG_CSV = "results/masking/grid_based_masking_debug_4_classes.csv"


# Directories for saving plots and individual images
PLOT_DIR = "plots/grid_based_masking_images"
GRID_BASED_ORIGINAL_DIR = "plots/grid_based_original"
GRID_BASED_MASKED_RECONSTRUCTED_DIR = "plots/grid_based_masked_reconstructed"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(GRID_BASED_ORIGINAL_DIR, exist_ok=True)
os.makedirs(GRID_BASED_MASKED_RECONSTRUCTED_DIR, exist_ok=True)

IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
def load_models():
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))

    encoder.eval()
    decoder.eval()
    classifier.eval()

    return encoder, decoder, classifier

# ------------------------------------------------------------------------------
# Transformation Pipeline
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
CLASS_LABELS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}

def calculate_image_metrics(original, modified):
    original_np = original.cpu().squeeze().numpy().transpose(1, 2, 0)
    modified_np = modified.cpu().squeeze().numpy().transpose(1, 2, 0)
    original_np = (original_np * 255).astype(np.uint8)
    modified_np = (modified_np * 255).astype(np.uint8)
    
    return {
        "SSIM": round(ssim(original_np, modified_np, channel_axis=-1, full=True)[0], 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }
    
def plot_and_save_images(input_image: torch.Tensor, 
                         reconstructed_image: torch.Tensor, 
                         masked_image: torch.Tensor, 
                         reconstructed_masked_image: torch.Tensor, 
                         filename: str) -> None:
    """
    Save a plot of the input image, reconstructed image, grid-masked image, 
    and reconstructed grid-masked image.

    Args:
        input_image (torch.Tensor): Original input image.
        reconstructed_image (torch.Tensor): Reconstructed image from the encoder-decoder.
        masked_image (torch.Tensor): Image after applying grid-based masking.
        reconstructed_masked_image (torch.Tensor): Reconstructed image from the masked image.
        filename (str): Path to save the plot.
    """
    # Convert tensors to numpy arrays for visualization
    input_np = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    masked_np = masked_image.cpu().squeeze().permute(1, 2, 0).numpy()

    # Resize reconstructed images to match the input image size
    input_size = input_image.size()[2:]  # (Height, Width)
    reconstructed_np = F.interpolate(reconstructed_image, size=input_size, mode="bilinear", align_corners=False).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    reconstructed_masked_np = F.interpolate(reconstructed_masked_image, size=input_size, mode="bilinear", align_corners=False).cpu().squeeze().permute(1, 2, 0).detach().numpy()

    # Create a subplot with 4 images
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(input_np)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(reconstructed_np)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")

    axs[2].imshow(masked_np)
    axs[2].set_title("Grid-Masked Image")
    axs[2].axis("off")

    axs[3].imshow(reconstructed_masked_np)
    axs[3].set_title("Reconstructed Masked Image")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    logging.info(f"Saved plot to: {filename}")

def apply_grid_mask(image, grid_size, pos):
    num_rows, num_cols = grid_size
    masked_image = image.clone()
    
    row_idx, col_idx = divmod(pos, num_cols)
    row_start = row_idx * (IMAGE_HEIGHT // num_rows)
    row_end = (row_idx + 1) * (IMAGE_HEIGHT // num_rows)
    col_start = col_idx * (IMAGE_WIDTH // num_cols)
    col_end = (col_idx + 1) * (IMAGE_WIDTH // num_cols)
    
    masked_image[:, :, row_start:row_end, col_start:col_end] = 0
    return masked_image

# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_grid_based_masking():
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(OUTPUT_CSV)
    
    encoder, decoder, classifier = load_models()

    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(TEST_DIR, image_filename)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue
        
        input_image = transform(image).unsqueeze(0).to(device)
        original_prediction = row["Prediction (Before Masking)"]
        
        counterfactual_found = False
        final_prediction = original_prediction
        confidence_final = None
        grid_size_found, grid_position_found = None, None
        metrics = {}

        with torch.no_grad():
            unchanged_positions = []  # Track positions where prediction remains the same
            ce_position = None  # Store the first position where CE occurs

            for grid_size in [(10, 5), (4, 2)]:  # Different grid sizes
                num_positions = grid_size[0] * grid_size[1]

                for pos in range(num_positions):
                    # Step 1: Apply Grid-Based Masking
                    masked_image = apply_grid_mask(input_image, grid_size, pos)

                    # Step 2: Encode Masked Image
                    latent_vector_masked = encoder(masked_image)[2]

                    # Step 3: Decode the Masked Latent Space
                    reconstructed_masked_image = decoder(latent_vector_masked)

                    # Step 3.5: Ensure the reconstructed image is correctly shaped
                    if reconstructed_masked_image.shape[1:] != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
                        reconstructed_masked_image = F.interpolate(reconstructed_masked_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)

                    # Step 4: Re-encode the Reconstructed Masked Image
                    latent_vector_re_encoded = encoder(reconstructed_masked_image)[2]

                    # Step 5: Classify the Re-Encoded Latent Space
                    masked_prediction = classifier(latent_vector_re_encoded)
                    confidence_final_tensor = F.softmax(masked_prediction, dim=1)
                    confidence_final = confidence_final_tensor.cpu().detach().numpy().flatten()

                    predicted_label_after_masking = torch.argmax(masked_prediction, dim=1).item()
                    final_prediction = CLASS_LABELS[predicted_label_after_masking]
                    
                    # save individual images
                    plot_and_save_images(input_image, reconstructed_masked_image, masked_image, reconstructed_masked_image,
                                            os.path.join(PLOT_DIR, f"{image_filename}_{grid_size}_{pos}.png"))
                    
                    # Convert tensors to PIL images for saving
                    input_pil = to_pil_image(input_image.squeeze())
                    masked_pil = to_pil_image(masked_image.squeeze())
                    reconstructed_pil = to_pil_image(reconstructed_masked_image.squeeze())
                    reconstructed_masked_pil = to_pil_image(reconstructed_masked_image.squeeze())
                    
                    input_pil.save(os.path.join(GRID_BASED_ORIGINAL_DIR, f"{image_filename}_{grid_size}_{pos}_original.png"))
                    masked_pil.save(os.path.join(GRID_BASED_ORIGINAL_DIR, f"{image_filename}_{grid_size}_{pos}_masked.png"))
                    reconstructed_pil.save(os.path.join(GRID_BASED_MASKED_RECONSTRUCTED_DIR, f"{image_filename}_{grid_size}_{pos}_reconstructed.png"))
                    reconstructed_masked_pil.save(os.path.join(GRID_BASED_MASKED_RECONSTRUCTED_DIR, f"{image_filename}_{grid_size}_{pos}_reconstructed_masked.png"))
                    

                    # Step 6: Compare Predictions
                    counterfactual_found = final_prediction != original_prediction
                    if counterfactual_found:
                        ce_position = pos  # Store the first CE position
                        grid_size_found, grid_position_found = str(grid_size), pos
                        metrics = calculate_image_metrics(input_image, reconstructed_masked_image)

                        logging.info(f"🔄 CE Found! Image: {image_filename}, Grid Size: {grid_size}, Position: {pos}, Prediction changed from {original_prediction} ➝ {final_prediction}")
                        break  # Stop checking further positions once CE is found
                    else:
                        unchanged_positions.append(pos)

                    # Save this result into the debug CSV
                    debug_entry = {
                        "Image File": image_filename,
                        "Grid Size": str(grid_size),
                        "Grid Position": pos,
                        "Prediction Before Masking": original_prediction,
                        "Prediction After Masking": final_prediction,
                        "Counterfactual Found": counterfactual_found
                    }
                    pd.DataFrame([debug_entry]).to_csv(DEBUG_CSV, mode="a", header=False, index=False)

                if counterfactual_found:
                    break  # Stop processing further once CE is found



        end_time = time.time()
        # Update the time: add the new time to the previous value if it exists.
        previous_time = row.get("Time Taken (s)", 0)
        total_time_taken = round(end_time - start_time + float(previous_time), 5)

        df_results.loc[df_results["Image File"] == image_filename, [
            "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found", 
            "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ]] = [
            final_prediction,
            ", ".join(map(str, confidence_final)) if confidence_final is not None else None,
            counterfactual_found,
            grid_size_found,
            grid_position_found,
            metrics.get("SSIM"), metrics.get("MSE"), metrics.get("PSNR"),
            metrics.get("UQI"), metrics.get("VIFP"),
            total_time_taken
        ]

        df_results.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Updated CSV for image {image_filename}")

    logging.info(f"Grid-based masking results saved to {OUTPUT_CSV}")

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    process_grid_based_masking()
