import os
import sys
import time
import logging
import torch
import warnings
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# Add the parent directory (where masking_utils.py is located) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))


# Import utilities from masking_utils
from masking_utils import (
    load_models, METHODS_RESULTS, METHODS_HEADERS,
    calculate_image_metrics, save_images, update_results_csv
)
# ------------------------------------------------------------------------------
# Suppress specific FutureWarning from torch.utils.checkpoint
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")

# Suppress specific FutureWarning from torch.cuda.amp.autocast
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.")
# Use a regex pattern to catch any message mentioning torch.cuda.amp.autocast
warnings.filterwarnings("ignore", message=".*torch\\.cuda\\.amp\\.autocast.*", category=FutureWarning)


# ----------------------------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# Paths and Constants
# ----------------------------------------------------------------------------
OUTPUT_CSV = METHODS_RESULTS["object_detection"]
TEST_DIR = "dataset/town7_dataset/test/"
IMAGE_DIRS = {
    "original": "plots/object_detection_original",
    "masked": "plots/object_detection_masked",
    "reconstructed": "plots/object_detection_reconstructed"
}
for dir_path in IMAGE_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------
# Load Models
# ----------------------------------------------------------------------------
encoder, decoder, classifier = load_models()
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, autoshape=True)

# ----------------------------------------------------------------------------
# Transformation Pipeline
# ----------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])

CLASS_LABELS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}
# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_object_detection_masking():
    image_files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".png"))
    if not os.path.exists(OUTPUT_CSV):
        logging.info(f"{OUTPUT_CSV} not found. Creating a new CSV file.")
        df_results = pd.DataFrame(columns=METHODS_HEADERS["object_detection"])
        df_results.to_csv(OUTPUT_CSV, index=False)  # Create an empty file with headers
    else:
        try:
            df_results = pd.read_csv(OUTPUT_CSV)
        except pd.errors.EmptyDataError:
            logging.warning(f"{OUTPUT_CSV} is empty. Initializing with headers.")
            df_results = pd.DataFrame(columns=METHODS_HEADERS["object_detection"])
            df_results.to_csv(OUTPUT_CSV, index=False)

    for image_filename in image_files:
        start_time = time.time()
        image_path = os.path.join(TEST_DIR, image_filename)

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue

        logging.info(f"Processing image: {image_filename}")
        input_image = transform(pil_image).unsqueeze(0).to(device)
        latent_vector = encoder(input_image)[2]
        original_prediction = classifier(latent_vector)
        predicted_label_before_masking = CLASS_LABELS[torch.argmax(original_prediction, dim=1).item()]
        confidence_before_masking = [round(float(x), 5) for x in F.softmax(original_prediction, dim=1).cpu().detach().numpy().flatten()]
        
        results = yolo_model(pil_image)
        detections = results.xyxy[0]
        counterfactual_found = False
        objects_detected = "None"
        bbox_info = ""
        metrics = {}
        
        predicted_label_after_masking = predicted_label_before_masking  # Default to original prediction
        confidence_after_masking = confidence_before_masking  # Default confidence values

        if detections.numel() == 0:
            logging.warning(f"⚠️ No objects detected in {image_filename}. Skipping masking.")
        else:
            for det in detections:
                x_min, y_min, x_max, y_max = map(int, det[:4])
                objects_detected = results.names[int(det[5])]
                bbox_info = f"({x_min}, {y_min}, {x_max}, {y_max})"

                masked_image = input_image.clone()
                masked_image[:, :, y_min:y_max, x_min:x_max] = 0

                latent_vector_masked = encoder(masked_image)[2]
                reconstructed_masked_image = decoder(latent_vector_masked)
                reconstructed_masked_image = F.interpolate(reconstructed_masked_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)
                latent_vector_re_encoded = encoder(reconstructed_masked_image)[2]
                masked_prediction = classifier(latent_vector_re_encoded)
                predicted_label_after_masking = CLASS_LABELS[torch.argmax(masked_prediction, dim=1).item()]
                confidence_after_masking = [round(float(x), 5) for x in F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()]

                if predicted_label_after_masking != predicted_label_before_masking:
                    counterfactual_found = True
                    metrics = calculate_image_metrics(input_image, reconstructed_masked_image)
                    save_images(
                        image_filename, 
                        input_image, 
                        masked_image, 
                        reconstructed_masked_image, IMAGE_DIRS
                    )
                break

        end_time = time.time()
        total_time_taken = round(end_time - start_time, 5)

        logging.info(f"Updating CSV for image {image_filename} with results: {predicted_label_after_masking}, {confidence_after_masking}, CE Found: {counterfactual_found}")

        update_results_csv(
            "object_detection", image_filename, {
                "Prediction (Before Masking)": predicted_label_before_masking,
                "Confidence (Before Masking)": "[" + ", ".join(f"{round(x, 5)}" for x in confidence_before_masking) + "]",
                "Prediction (After Masking)": predicted_label_after_masking,
                "Confidence (After Masking)": "[" + ", ".join(f"{round(x, 5)}" for x in confidence_after_masking) + "]",
                "Counterfactual Found": counterfactual_found,
                "Grid Size": bbox_info,
                "Grid Position": bbox_info,
                "SSIM": metrics.get("SSIM", ""),
                "MSE": metrics.get("MSE", ""),
                "PSNR": metrics.get("PSNR", ""),
                "UQI": metrics.get("UQI", ""),
                "VIFP": metrics.get("VIFP", ""),
                "Objects Detected": objects_detected,
                "Time Taken (s)": total_time_taken
            }, OUTPUT_CSV
        )

        logging.info(f"Updated CSV for image {image_filename}")

if __name__ == "__main__":
    process_object_detection_masking()