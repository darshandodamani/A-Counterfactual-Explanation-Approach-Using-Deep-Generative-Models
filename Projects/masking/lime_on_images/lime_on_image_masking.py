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
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

# Add the parent directory (where masking_utils.py is located) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

# Import utilities from masking_utils
from masking_utils import (
    load_models, METHODS_RESULTS, METHODS_HEADERS,
    calculate_image_metrics, save_images, update_results_csv
)

# ----------------------------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------------------------------------------------------
# Paths and Constants
# ----------------------------------------------------------------------------
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS_4_CLASS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}
CLASS_LABELS_2_CLASS = {0: "STOP", 1: "GO"}

# ----------------------------------------------------------------------------
# Run for both 2_class and 4_class
# ----------------------------------------------------------------------------
def process_lime_on_image_masking_all():
    for classifier_type in ["2_class", "4_class"]:
        logging.info(f"Running LIME on Image Masking for {classifier_type} classification")

        encoder, decoder, classifier = load_models(classifier_type)
        logging.info(f"Models loaded for {classifier_type}")

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

        TEST_DIR = "dataset/town7_dataset/test/"

        for image_filename in sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".png")):
            start_time = time.time()
            image_path = os.path.join(TEST_DIR, image_filename)

            try:
                pil_image = Image.open(image_path).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            except Exception as e:
                logging.error(f"Error loading image {image_path}: {e}")
                continue

            logging.info(f"Processing image: {image_filename}")
            input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

            latent_vector = encoder(input_image)[2]
            original_prediction = classifier(latent_vector)
            predicted_label_before_masking_idx = torch.argmax(original_prediction, dim=1).item()
            predicted_label_before_masking = label_mapping[predicted_label_before_masking_idx]
            confidence_before_masking = str([round(float(x), 5) for x in F.softmax(original_prediction, dim=1).cpu().detach().numpy().flatten()])

            explainer = LimeImageExplainer()

            def predict_fn(images):
                images_tensor = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                images_tensor = F.interpolate(images_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)

                with torch.no_grad():
                    latent_vectors = encoder(images_tensor)[2]
                    predictions = classifier(latent_vectors)
                    probabilities = F.softmax(predictions, dim=1).cpu().numpy().astype(np.float32)

                    if np.isnan(probabilities).any():
                        logging.warning(" NaN detected in softmax output! Replacing with uniform probabilities.")
                        num_classes = probabilities.shape[1]
                        probabilities = np.ones_like(probabilities) / num_classes

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
                continue

            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=True)

            masked_np = mark_boundaries(temp / 255.0, mask)
            masked_pil = Image.fromarray((masked_np * 255).astype(np.uint8)).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            masked_tensor = transforms.ToTensor()(masked_pil).unsqueeze(0).to(device)

            latent_vector_masked = encoder(masked_tensor)[2]
            reconstructed_masked_image = decoder(latent_vector_masked)
            reconstructed_masked_image = F.interpolate(reconstructed_masked_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)

            latent_vector_re_encoded = encoder(reconstructed_masked_image)[2]
            masked_prediction = classifier(latent_vector_re_encoded)
            predicted_label_after_masking_idx = torch.argmax(masked_prediction, dim=1).item()
            predicted_label_after_masking = label_mapping[predicted_label_after_masking_idx]
            confidence_after_masking = str([round(float(x), 5) for x in F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()])

            counterfactual_found = predicted_label_after_masking != predicted_label_before_masking
            metrics = calculate_image_metrics(input_image, reconstructed_masked_image) if counterfactual_found else {}

            save_images(image_filename, input_image, masked_tensor, reconstructed_masked_image, IMAGE_DIRS)

            end_time = time.time()
            total_time_taken = round(end_time - start_time, 5)

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

            logging.info(f" Updated CSV for image {image_filename} ({classifier_type})")

if __name__ == "__main__":
    process_lime_on_image_masking_all()