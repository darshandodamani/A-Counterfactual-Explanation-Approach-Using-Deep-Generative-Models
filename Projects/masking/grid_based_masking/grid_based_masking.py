# location: Projects/masking/grid_based_masking/grid_based_masking.py
import os
import sys
import time
import logging
import torch
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# Add the parent directory (where masking_utils.py is located) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

# Import utilities from masking_utils
from masking_utils import (
    load_models, METHODS_RESULTS, METHODS_HEADERS,
    calculate_image_metrics, save_images, update_results_csv
)

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------------------------------------------------------
# Paths and Constants
# ----------------------------------------------------------------------------
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS_4_CLASS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}
CLASS_LABELS_2_CLASS = {0: "STOP", 1: "GO"}

def apply_grid_mask(image, grid_size, pos):
    """
    Applies grid-based masking by blacking out a specific grid cell.
    """
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
def process_grid_based_masking(classifier_type: str = "4_class"):
    """
    Runs grid-based masking for either 2_class or 4_class classification.

    Args:
        classifier_type (str): "2_class" or "4_class" to specify which classifier to use.
    """
    logging.info(f"Starting grid-based masking with {classifier_type} classification.")

    # Load the models
    encoder, decoder, classifier = load_models(classifier_type)
    logging.info(f"Models loaded for {classifier_type}.")

    # Select the correct CSV file and label mapping based on classifier type
    output_csv = METHODS_RESULTS[f"grid_based_{classifier_type}"]
    logging.info(f"Output CSV: {output_csv}")

    label_mapping = CLASS_LABELS_2_CLASS if classifier_type == "2_class" else CLASS_LABELS_4_CLASS

    IMAGE_DIRS = {
        "original": f"plots/grid_based_{classifier_type}_original",
        "masked": f"plots/grid_based_{classifier_type}_masked",
        "reconstructed": f"plots/grid_based_{classifier_type}_reconstructed"
    }

    for dir_path in IMAGE_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

    # Initialize CSV if it does not exist
    if not os.path.exists(output_csv):
        logging.info(f"{output_csv} not found. Creating a new CSV file.")
        df_results = pd.DataFrame(columns=METHODS_HEADERS[f"grid_based_{classifier_type}"])
        df_results.to_csv(output_csv, index=False)
    else:
        try:
            df_results = pd.read_csv(output_csv)
        except pd.errors.EmptyDataError:
            logging.warning(f"{output_csv} is empty. Initializing with headers.")
            df_results = pd.DataFrame(columns=METHODS_HEADERS[f"grid_based_{classifier_type}"])
            df_results.to_csv(output_csv, index=False)

    # Image directory
    TEST_DIR = "dataset/town7_dataset/test/"

    for image_filename in sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".png")):
        start_time = time.time()
        image_path = os.path.join(TEST_DIR, image_filename)

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue

        logging.info(f"Processing image: {image_filename}")
        input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

        # Step 1: Get Initial Prediction
        latent_vector = encoder(input_image)[2]
        original_prediction = classifier(latent_vector)
        predicted_label_before_masking_idx = torch.argmax(original_prediction, dim=1).item()
        predicted_label_before_masking = label_mapping[predicted_label_before_masking_idx]
        confidence_before_masking = [round(float(x), 5) for x in F.softmax(original_prediction, dim=1).cpu().detach().numpy().flatten()]

        # Step 2: Apply Grid-Based Masking (Early Stopping Optimization)
        counterfactual_found = False  # Track if CE is found

        for grid_size in [(10, 5), (4, 2)]:  # First finer grid, then coarser grid
            if counterfactual_found:
                break  # Stop checking if CE was already found

            num_rows, num_cols = grid_size
            total_positions = num_rows * num_cols

            for pos in range(total_positions):
                masked_image = apply_grid_mask(input_image, grid_size, pos)

                # Encode Masked Image
                latent_vector_masked = encoder(masked_image)[2]

                # Decode and Resize Image
                reconstructed_masked_image = decoder(latent_vector_masked)
                reconstructed_masked_image = F.interpolate(
                    reconstructed_masked_image,
                    size=(input_image.shape[2], input_image.shape[3]),
                    mode="bilinear",
                    align_corners=False
                )

                # Re-encode and Classify
                latent_vector_re_encoded = encoder(reconstructed_masked_image)[2]
                masked_prediction = classifier(latent_vector_re_encoded)

                # Get Predicted Label After Masking
                predicted_label_after_masking_idx = torch.argmax(masked_prediction, dim=1).item()
                predicted_label_after_masking = label_mapping[predicted_label_after_masking_idx]

                # If classification changes, CE is found
                if predicted_label_after_masking != predicted_label_before_masking:
                    counterfactual_found = True  # Mark CE as found
                    metrics = calculate_image_metrics(input_image, reconstructed_masked_image)

                    # Save Images Only if Counterfactual is Found
                    save_images(image_filename, input_image, masked_image, reconstructed_masked_image, IMAGE_DIRS)
                    break  # Stop checking more positions within this grid size

        # Step 3: Calculate Time Taken
        end_time = time.time()
        total_time_taken = round(end_time - start_time, 5)

        # Step 4: **Ensure Separate CSV Update for 2-Class and 4-Class**
        update_results_csv(
            f"grid_based_{classifier_type}",
            image_filename, {
                "Prediction (Before Masking)": predicted_label_before_masking,
                "Confidence (Before Masking)": confidence_before_masking,
                "Prediction (After Masking)": predicted_label_after_masking,
                "Confidence (After Masking)": confidence_before_masking,
                "Counterfactual Found": counterfactual_found,
                "Grid Size": f"{grid_size}" if counterfactual_found else "N/A",
                "Grid Position": f"{pos}" if counterfactual_found else "N/A",
                "SSIM": metrics.get("SSIM", "") if counterfactual_found else "",
                "MSE": metrics.get("MSE", "") if counterfactual_found else "",
                "PSNR": metrics.get("PSNR", "") if counterfactual_found else "",
                "UQI": metrics.get("UQI", "") if counterfactual_found else "",
                "VIFP": metrics.get("VIFP", "") if counterfactual_found else "",
                "Time Taken (s)": total_time_taken
            }, output_csv
        )

        logging.info(f" Updated CSV for image {image_filename} (Classifier: {classifier_type})")

if __name__ == "__main__":
    process_grid_based_masking()
