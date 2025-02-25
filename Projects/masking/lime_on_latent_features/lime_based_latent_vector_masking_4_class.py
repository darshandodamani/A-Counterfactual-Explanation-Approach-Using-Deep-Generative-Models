#location: Projects/masking/lime_on_latent_features/lime_based_latent_vector_masking_4_class.py
import os
import sys
import time
import logging
import csv
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from lime.lime_tabular import LimeTabularExplainer
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Tuple

# ------------------------------------------------------------------------------
# Setup Python Paths for local modules
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifiers_4_class import ClassifierModel as Classifier

sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
from initialize_masking_pipeline import INITIAL_PREDICTIONS_CSV as initial_predictions_csv

# ------------------------------------------------------------------------------
# Configuration and Paths
# ------------------------------------------------------------------------------
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_4_classes.pth"
MEDIAN_VALUES_CSV = "latent_vectors/combined_median_values.csv"

OUTPUT_CSV = "results/masking/lime_on_latent_masking_4_classes_results.csv"
REPLACEMENTS_CSV = "results/masking/lime_on_latent_replacements_4_classes.csv"
TEST_DIR = "dataset/town7_dataset/test/"

# Directories for saving images
IMAGE_DIRS = {
    "original": "plots/lime_on_latent_original",
    "masked": "plots/lime_on_latent_masked",
    "reconstructed": "plots/lime_on_latent_reconstructed",
    "difference": "plots/lime_on_latent_difference"
}

for dir_path in IMAGE_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)
    
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
CLASS_LABELS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}

def save_image(tensor, path):
    """Converts tensor to PIL image and saves it."""
    img = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_pil.save(path)

# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
def load_models() -> Tuple[VariationalEncoder, Decoder, Classifier]:
    """
    Loads the encoder, decoder, and classifier models onto the specified device.
    """
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    classifier.eval()

    logging.info("Models loaded and set to evaluation mode.")
    return encoder, decoder, classifier

# Load the models
global encoder, decoder, classifier
encoder, decoder, classifier = load_models()

# ------------------------------------------------------------------------------
# Load Auxiliary Data
# ------------------------------------------------------------------------------
median_values = pd.read_csv(MEDIAN_VALUES_CSV).values.flatten()
df_initial = pd.read_csv(initial_predictions_csv)
df_results = pd.read_csv(OUTPUT_CSV)

# ------------------------------------------------------------------------------
# Transformations
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def calculate_image_metrics(original: np.ndarray, modified: np.ndarray) -> Dict[str, float]:
    """Computes SSIM, MSE, PSNR, UQI, and VIFP between two images."""
    return {
        "SSIM": round(ssim(original, modified, data_range=1.0, channel_axis=-1), 5),
        "MSE": round(mse(original, modified), 5),
        "PSNR": round(psnr(original, modified, data_range=1.0), 5),
        "UQI": round(uqi(original, modified), 5),
        "VIFP": round(vifp(original, modified), 5),
    }

def predict_with_latent(latent: np.ndarray) -> np.ndarray:
    """Returns prediction probabilities for a given latent vector."""
    latent_tensor = torch.tensor(latent, dtype=torch.float32).to(device)
    output = classifier(latent_tensor)
    return F.softmax(output, dim=1).cpu().detach().numpy()

# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_lime_on_latent_masking() -> None:
    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(TEST_DIR, image_filename)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue
        
        # Step 1: Convert image to tensor and obtain latent representation
        input_image = transforms.ToTensor()(image).unsqueeze(0).to(device)
        
        # Read the original prediction from the CSV
        original_prediction = str(row["Prediction (Before Masking)"]).strip()
        logging.info(f"Original Prediction from CSV: {original_prediction}")
        
        confidence_values = np.array(eval(row["Confidence (Before Masking)"]), dtype=np.float32)
        label = np.argmax(confidence_values)
        counterfactual_found = False
        confidence_final_str = ""
        selected_features_str = ""
        metrics = {}  # Dictionary to store image quality metrics

        # Step 2: Encode image to obtain latent vector
        latent_vector = encoder(input_image)[2]
        latent_vector_np = latent_vector.cpu().detach().numpy().reshape(1, -1)

        # Step 3: Set up LIME explainer on latent features
        explainer = LimeTabularExplainer(
            latent_vector_np,
            mode="classification",
            feature_names=[f"latent_{i}" for i in range(latent_vector_np.shape[1])],
            discretize_continuous=False
        )
        
        explanation = explainer.explain_instance(
            latent_vector_np.flatten(), predict_with_latent, num_features=latent_vector_np.shape[1]
        )

        # Step 4: Identify negatively influential features and apply masking
        important_features = sorted(
            [(int(feature.split("_")[-1]), weight) for feature, weight in explanation.as_list() if weight > 0],
            key=lambda x: abs(x[1]), reverse=True
        )
        
        for feature_index, _ in important_features:
            masked_latent_vector = latent_vector_np.flatten().copy()
            masked_latent_vector[feature_index] = median_values[feature_index]
            
            # Step 5: Decode the masked latent vector to reconstruct the image
            masked_latent_tensor = torch.tensor(masked_latent_vector, dtype=torch.float32).to(device).reshape(1, -1)
            reconstructed_image = decoder(masked_latent_tensor)
            
            # Ensure reconstructed image is resized before encoding
            if reconstructed_image.shape[1:] != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
                reconstructed_image = F.interpolate(reconstructed_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)
                            
            # Flatten the image before sending to the encoder
            reconstructed_image = reconstructed_image.view(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
            
            # Step 6: Re-encode the reconstructed image
            latent_vector_re_encoded = encoder(reconstructed_image)[2]
            
            # Step 7: Classify the re-encoded latent vector
            masked_prediction = classifier(latent_vector_re_encoded)
            confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
            confidence_final_str = "[" + ", ".join(f"{x:.5f}" for x in confidence_final) + "]"
            final_prediction = CLASS_LABELS[torch.argmax(masked_prediction, dim=1).item()].strip()
            logging.info(f"Final Prediction after Masking: '{final_prediction}'")

            # Step 8: Ensure CE is only found when prediction changes
            logging.info(f"Original Prediction: {original_prediction}, Final Prediction: {final_prediction}")
            if final_prediction.strip() != original_prediction.strip():
                counterfactual_found = True  # Counterfactual is found if predictions are different
                selected_features_str = str(feature_index)

                # Step 8.1: Save Images Only When Counterfactual is Found (CE=True)
                save_image(input_image, os.path.join(IMAGE_DIRS["masked"], image_filename))
                save_image(reconstructed_image, os.path.join(IMAGE_DIRS["reconstructed"], image_filename))
                save_image((input_image - reconstructed_image).abs(), os.path.join(IMAGE_DIRS["difference"], image_filename))
                
                # Step 8.2: Calculate Image Metrics
                original_np = input_image.cpu().squeeze().numpy().transpose(1, 2, 0)  # Convert original to NumPy
                reconstructed_np = reconstructed_image.cpu().squeeze().detach().numpy().transpose(1, 2, 0)  # Convert reconstructed to NumPy

                metrics = {
                    "SSIM": round(ssim(original_np, reconstructed_np, data_range=1.0, channel_axis=-1), 5),
                    "MSE": round(mse(original_np, reconstructed_np), 5),
                    "PSNR": round(psnr(original_np, reconstructed_np, data_range=1.0), 5),
                    "UQI": round(uqi(original_np, reconstructed_np), 5),
                    "VIFP": round(vifp(original_np, reconstructed_np), 5),
                }
            else:
                counterfactual_found = False  # Counterfactual is not found if predictions match
                selected_features_str = ""    # Clear selected features if no CE is found
                metrics = {"SSIM": "", "MSE": "", "PSNR": "", "UQI": "", "VIFP": ""}  # Clear metrics if no CE

        # Step 9: Calculate total time taken
        end_time = time.time()
        previous_time = float(row.get("Time Taken (s)", 0) or 0)  # Fixed conversion issue
        total_time_taken = round(end_time - start_time + previous_time, 5)
        
        # Step 10: Update results CSV
        df_results.loc[df_results['Image File'] == image_filename, [
            "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
            "Selected Features", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ]] = [
            final_prediction,
            confidence_final_str,
            counterfactual_found,
            selected_features_str,
            metrics.get("SSIM", ""),  # Include metrics
            metrics.get("MSE", ""),
            metrics.get("PSNR", ""),
            metrics.get("UQI", ""),
            metrics.get("VIFP", ""),
            total_time_taken
        ]
        
        df_results.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Updated CSV for image {image_filename}: Counterfactual Found = {counterfactual_found}, Time Taken = {total_time_taken}s")
        
if __name__ == "__main__":
    process_lime_on_latent_masking()

