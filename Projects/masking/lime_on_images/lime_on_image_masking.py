# location: Projects/masking/lime_on_images/lime_on_image_masking.py
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
from lime import lime_image
from skimage.segmentation import mark_boundaries

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

# ------------------------------------------------------------------------------
# LIME Helper Function
# ------------------------------------------------------------------------------
def classifier_fn(images, encoder, classifier):
    """LIME requires a function that takes a batch of images and returns predictions."""
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  
    images = F.interpolate(images, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)

    with torch.no_grad():
        latent_vectors = encoder(images)[2]  # Get latent space representation
        predictions = classifier(latent_vectors)
    
    return predictions.cpu().detach().numpy()

def apply_lime_mask(image: torch.Tensor, mask: np.ndarray, alpha: float = 0.5) -> torch.Tensor:
    """
    Applies LIME mask with alpha blending instead of hard masking.

    Args:
        image (torch.Tensor): Original image tensor (C, H, W).
        mask (np.ndarray): Mask from LIME (H, W) - binary or soft.
        alpha (float): Blending factor (0.0 = no mask, 1.0 = fully masked).

    Returns:
        torch.Tensor: Blended masked image.
    """
    # Convert the mask to a tensor and expand dimensions to match the image
    mask_tensor = torch.from_numpy(mask).float().to(image.device)  # Convert mask to tensor
    mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)
    
    # Define blending background (e.g., black or blurred)
    background = torch.zeros_like(image)  # Black background
    
    # Alpha blend: (1 - alpha) * image + alpha * background
    blended_image = (1 - alpha) * image + alpha * background * mask_tensor

    return blended_image


# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_lime_on_image_masking(classifier_type: str = "4_class"):
    logging.info(f"Running LIME on Image Masking for {classifier_type} classification")
    
    # Load the models
    encoder, decoder, classifier = load_models(classifier_type)
    logging.info(f"Models loaded for {classifier_type}")
    
    # Select the correct CSV file and label mapping based on classifier type
    output_csv = METHODS_RESULTS[f"lime_on_image_{classifier_type}"]
    logging.info(f"Output CSV file: {output_csv}")
    
    label_mapping = CLASS_LABELS_2_CLASS if classifier_type == "2_class" else CLASS_LABELS_4_CLASS
    
    IMAGE_DIRS = {
        "original": f"plots/lime_on_image_{classifier_type}_original",
        "masked": f"plots/lime_on_image_{classifier_type}_masked",
        "reconstructed": f"plots/lime_on_image_{classifier_type}_reconstructed"
    }
    
    for dir_name in IMAGE_DIRS.values():
        os.makedirs(dir_name, exist_ok=True)
        
    # Initialize CSV if it does not exist
    if not os.path.exists(output_csv):
        logging.info(f"{output_csv} not found. Creating a new CSV file.")
        df_results = pd.DataFrame(columns=METHODS_HEADERS[f"lime_on_image_{classifier_type}"])
        df_results.to_csv(output_csv, index=False)
    else:
        try:
            df_results = pd.read_csv(output_csv)
        except pd.errors.EmptyDataError:
            logging.warning(f"{output_csv} is empty. Initializing with headers.")
            df_results = pd.DataFrame(columns=METHODS_HEADERS[f"lime_on_image_{classifier_type}"])
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
        confidence_before_masking = str([round(float(x), 5) for x in F.softmax(original_prediction, dim=1).cpu().detach().numpy().flatten()])

        # Step 2: Perform LIME on the image
        explainer = lime_image.LimeImageExplainer()

        def predict_fn(images):
            """Predict function for LIME, ensuring no NaNs in softmax probabilities."""
            images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            with torch.no_grad():
                latent_vectors = encoder(images_tensor)[2]
                predictions = classifier(latent_vectors)

                # Convert logits to probabilities safely
                probabilities = F.softmax(predictions, dim=1)
                probabilities = probabilities.cpu().numpy().astype(np.float32)  # Ensure float32

                # Check for NaNs and replace them
                if np.isnan(probabilities).any():
                    logging.warning("⚠️ NaN detected in softmax output! Replacing with uniform probabilities.")
                    num_classes = probabilities.shape[1]
                    probabilities = np.ones_like(probabilities) / num_classes  # Uniform distribution

            return probabilities


        try:
            explanation = explainer.explain_instance(
                np.array(pil_image),
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )
        except ValueError as e:
            logging.error(f"LIME failed due to NaNs: {e}")
            continue  # Skip the image if LIME fails

        # Get the mask
        top_label = explanation.top_labels[0]
        mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5)[1]

        # Step 3: Apply Alpha Blending
        masked_image = apply_lime_mask(input_image, mask, alpha=0.5)

        # Step 4: Encode Masked Image
        latent_vector_masked = encoder(masked_image)[2]

        # Step 5: Decode Masked Latent Space
        reconstructed_masked_image = decoder(latent_vector_masked)
        reconstructed_masked_image = F.interpolate(reconstructed_masked_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)

        # Step 6: Re-encode the Reconstructed Masked Image
        latent_vector_re_encoded = encoder(reconstructed_masked_image)[2]

        # Step 7: Classify the Re-Encoded Latent Space
        masked_prediction = classifier(latent_vector_re_encoded)
        predicted_label_after_masking_idx = torch.argmax(masked_prediction, dim=1).item()
        predicted_label_after_masking = label_mapping[predicted_label_after_masking_idx]
        confidence_after_masking = str([round(float(x), 5) for x in F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()])

        # Step 8: Compute Image Metrics and Save Images
        counterfactual_found = predicted_label_after_masking != predicted_label_before_masking
        metrics = calculate_image_metrics(input_image, reconstructed_masked_image) if counterfactual_found else {}

        save_images(image_filename, input_image, masked_image, reconstructed_masked_image, IMAGE_DIRS)

        # Step 9: Calculate Processing Time
        end_time = time.time()
        total_time_taken = round(end_time - start_time, 5)

        # Step 10: Update CSV
        update_results_csv(
            f"lime_on_image_{classifier_type}",
            image_filename, {
                "Prediction (Before Masking)": predicted_label_before_masking,
                "Confidence (Before Masking)": confidence_before_masking,
                "Prediction (After Masking)": predicted_label_after_masking,
                "Confidence (After Masking)": confidence_after_masking,
                "Counterfactual Found": counterfactual_found,
                "SSIM": metrics.get("SSIM", ""),
                "MSE": metrics.get("MSE", ""),
                "PSNR": metrics.get("PSNR", ""),
                "UQI": metrics.get("UQI", ""),
                "VIFP": metrics.get("VIFP", ""),
                "Time Taken (s)": total_time_taken
            }, output_csv
        )

        logging.info(f" Updated CSV for image {image_filename}")

if __name__ == "__main__":
    process_lime_on_image_masking()
