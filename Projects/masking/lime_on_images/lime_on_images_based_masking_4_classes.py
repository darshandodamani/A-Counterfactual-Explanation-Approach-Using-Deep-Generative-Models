import os
import sys
import time
import ast
import logging
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# Add Python Paths for Local Modules
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "autoencoder")))
from encoder import VariationalEncoder
from decoder import Decoder
from classifiers_4_class import ClassifierModel as Classifier

sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
from initialize_masking_pipeline_4_class import INITIAL_PREDICTIONS_CSV as initial_predictions_csv

# ------------------------------------------------------------------------------
# Paths and Constants
# ------------------------------------------------------------------------------
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_4_classes.pth"
OUTPUT_CSV = "results/masking/lime_on_image_masking_4_classes_results.csv"
TEST_DIR = "dataset/town7_dataset/test/"

# Directories for saving images
IMAGE_DIRS = {
    "original": "plots/lime_on_image_original",
    "masked": "plots/lime_on_image_masked",
    "reconstructed": "plots/lime_on_image_reconstructed",
    "difference": "plots/lime_on_image_difference"
}

for dir_path in IMAGE_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Mapping
CLASS_LABELS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}

# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
def load_models():
    """Loads encoder, decoder, and classifier models and sets them to evaluation mode."""
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    classifier.eval()

    return encoder, decoder, classifier

# ------------------------------------------------------------------------------
# Image Quality Metrics Calculation
# ------------------------------------------------------------------------------
def calculate_image_metrics(original: torch.Tensor, modified: torch.Tensor) -> Dict[str, float]:
    """Computes SSIM, MSE, PSNR, UQI, and VIFP between two images."""
    original_np = original.cpu().squeeze().numpy().transpose(1, 2, 0)
    modified_np = modified.cpu().squeeze().numpy().transpose(1, 2, 0)

    original_np = (original_np * 255).astype(np.uint8)
    modified_np = (modified_np * 255).astype(np.uint8)

    return {
        "SSIM": round(ssim(original_np, modified_np, channel_axis=-1, data_range=255), 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np, data_range=255), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }
    
def save_images(image_filename, input_image, masked_image, reconstructed_masked_image):
    """Saves original, masked, reconstructed, and difference images when CE is found."""
    base_filename, _ = os.path.splitext(image_filename)

    # Convert tensors to PIL images
    original_pil = to_pil_image(input_image.squeeze(0).cpu())
    masked_pil = to_pil_image(masked_image.squeeze(0).cpu())
    reconstructed_pil = to_pil_image(reconstructed_masked_image.squeeze(0).cpu())
    
    # Compute difference image
    difference_image = torch.abs(input_image - reconstructed_masked_image)
    difference_pil = to_pil_image(difference_image.squeeze(0).cpu())

    # Save images in their respective directories
    original_pil.save(os.path.join(IMAGE_DIRS["original"], f"{base_filename}_original.png"))
    masked_pil.save(os.path.join(IMAGE_DIRS["masked"], f"{base_filename}_masked.png"))
    reconstructed_pil.save(os.path.join(IMAGE_DIRS["reconstructed"], f"{base_filename}_reconstructed.png"))
    difference_pil.save(os.path.join(IMAGE_DIRS["difference"], f"{base_filename}_difference.png"))

    logging.info(f"✅ Saved images for {image_filename}")


# ------------------------------------------------------------------------------
# LIME Classifier Prediction Function
# ------------------------------------------------------------------------------
def classifier_prediction(image_np: np.ndarray) -> np.ndarray:
    """
    Prediction function for LIME. Converts a numpy image (H, W, C) to tensor, passes it through
    the encoder and classifier, and returns prediction probabilities.
    """
    # Rearrange to (N, C, H, W)
    image_tensor = torch.tensor(image_np.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        latent_vector = encoder(image_tensor)[2]
        prediction = classifier(latent_vector)
        probabilities = F.softmax(prediction, dim=1).cpu().detach().numpy()
    
    # Ensure no NaN values exist in the probabilities
    probabilities = np.nan_to_num(probabilities, nan=0.0)

    return probabilities


# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_lime_on_image_masking():
    """Runs LIME-based masking for 4-class classification and updates CSV files."""
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(OUTPUT_CSV)
    global encoder, decoder, classifier
    encoder, decoder, classifier = load_models()

    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(TEST_DIR, image_filename)

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue

        input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

        # Step 1: Get Initial Prediction
        original_prediction = row["Prediction (Before Masking)"]
        counterfactual_found = False
        final_prediction = original_prediction
        confidence_final_str = ""
        metrics = {}

        # Step 2: Generate LIME Explanation & Apply Mask
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(pil_image), classifier_prediction, hide_color=0, num_samples=500
        )

        # Convert confidence string to a proper list
        confidence_values = np.array(ast.literal_eval(row["Confidence (Before Masking)"]), dtype=np.float32)

        # Get the predicted label
        label = np.argmax(confidence_values)

        temp, mask = explanation.get_image_and_mask(label=label, positive_only=True, num_features=10, hide_rest=False)

        # Convert the mask to match image dimensions (from [H, W] → [1, 3, H, W])
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

        # Apply LIME mask to all RGB channels correctly
        masked_image = input_image.clone()
        masked_image[mask == 0] = 0  # ✅ Now correctly applies mask to all 3 channels



        with torch.no_grad():
            # Step 3: Encode Masked Image
            latent_vector_masked = encoder(masked_image)[2]

            # Step 4: Decode the Masked Latent Space
            reconstructed_masked_image = decoder(latent_vector_masked)
            
            # Ensure reconstructed masked image is resized before re-encoding
            if reconstructed_masked_image.shape[1:] != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
                reconstructed_masked_image = F.interpolate(reconstructed_masked_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)

            # Step 5: Re-Encode the Reconstructed Masked Image
            latent_vector_re_encoded = encoder(reconstructed_masked_image)[2]

            # Step 6: Classify the Re-Encoded Latent Space
            masked_prediction = classifier(latent_vector_re_encoded)

            confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
            confidence_final_str = "[" + ", ".join(f"{x:.5f}" for x in confidence_final) + "]"
            predicted_label_after_masking = torch.argmax(masked_prediction, dim=1).item()
            final_prediction = CLASS_LABELS[predicted_label_after_masking]

        # Step 7: Check if Counterfactual Found
        counterfactual_found = final_prediction != row["Prediction (Before Masking)"]

        if counterfactual_found:
            # Step 7.1: Compute image quality metrics
            metrics = calculate_image_metrics(input_image, reconstructed_masked_image)

            # Step 7.2: Save images only if Counterfactual Explanation (CE) is found
            save_images(image_filename, input_image, masked_image, reconstructed_masked_image)

        total_time_taken = round(time.time() - start_time + float(row.get("Time Taken (s)", 0)), 5)

        # Step 8: Update CSV
        df_results.loc[df_results["Image File"] == image_filename, [
            "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
            "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ]] = [
            final_prediction,
            confidence_final_str,
            counterfactual_found,
            "Superpixel",
            "",  # Keep empty string for grid position since it's not relevant
            metrics.get("SSIM", np.nan),  # Use NaN instead of empty string for numerical columns
            metrics.get("MSE", np.nan),
            metrics.get("PSNR", np.nan),
            metrics.get("UQI", np.nan),
            metrics.get("VIFP", np.nan),
            total_time_taken  # Keep time as a float
        ]

        # ✅ Save CSV (ensures NaNs are written correctly)
        df_results.to_csv(OUTPUT_CSV, index=False)


    logging.info(f"LIME-based masking results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_lime_on_image_masking()
