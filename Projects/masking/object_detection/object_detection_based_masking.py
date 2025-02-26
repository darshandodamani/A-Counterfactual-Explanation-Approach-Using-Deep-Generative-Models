# location : Projects/masking/object_detection/object_detection_based_masking_4_classes.py
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
warnings.filterwarnings("ignore", category=FutureWarning, message="torch.cuda.amp.autocast(args...) is deprecated. Please use torch.amp.autocast('cuda', args...) instead.")
# Use a regex pattern to catch any message mentioning torch.cuda.amp.autocast
warnings.filterwarnings("ignore", message=".*torch\\.cuda\\.amp\\.autocast.*", category=FutureWarning)


# ----------------------------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# Paths and Constants
# ----------------------------------------------------------------------------
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS_4_CLASS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}
CLASS_LABELS_2_CLASS = {0: "STOP", 1: "GO"}

# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_object_detection_masking(classifier_type:str = "4_class"):
    """
    Runs object detection-based masking for either 2_class or 4_class classification.
    
    Args:
        classifier_type (str): "2_class" or "4_class" to specify which classifier to use.
    """
    logging.info(f" Starting object detection-based masking with {classifier_type} classification.")

    # Load correct models based on classifier type
    encoder, decoder, classifier = load_models(classifier_type)
    logging.info(f" Models loaded for {classifier_type}.")
    
    # select the csv file based on the classifier type
    output_csv = METHODS_RESULTS[f"object_detection_{classifier_type}"]
    logging.info(f" Output CSV: {output_csv}")
    
    # Select the correct label mapping based on classifier type
    label_mapping = CLASS_LABELS_2_CLASS if classifier_type == "2_class" else CLASS_LABELS_4_CLASS
    
    IMAGE_DIRS = {
        "original": f"plots/object_detection_{classifier_type}_original",
        "masked": f"plots/object_detection_{classifier_type}_masked",
        "reconstructed": f"plots/object_detection_{classifier_type}_reconstructed"
    }
    
    for dir_path in IMAGE_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Load YOLO model for object detection
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, autoshape=True)
    
    if not os.path.exists(output_csv):
        logging.info(f"{output_csv} not found. Creating a new CSV file.")
        df_results = pd.DataFrame(columns=METHODS_HEADERS[f"object_detection_{classifier_type}"])
        df_results.to_csv(output_csv, index=False)
    else:
        try:
            df_results = pd.read_csv(output_csv)
        except pd.errors.EmptyDataError:
            logging.warning(f"{output_csv} is empty. Initializing with headers.")
            df_results = pd.DataFrame(columns=METHODS_HEADERS[f"object_detection_{classifier_type}"])
            df_results.to_csv(output_csv, index=False)
            
    TEST_DIR = "dataset/town7_dataset/test/"
    for image_filename in sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".png")):
        start_time = time.time()
        image_path = os.path.join(TEST_DIR, image_filename)

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue

        logging.info(f" Processing image: {image_filename}")
        input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

        # Step 1: Get Initial Prediction
        latent_vector = encoder(input_image)[2]
        original_prediction = classifier(latent_vector)
        predicted_label_before_masking_idx = torch.argmax(original_prediction, dim=1).item()
        predicted_label_before_masking = label_mapping[predicted_label_before_masking_idx]
        confidence_before_masking = [round(float(x), 5) for x in F.softmax(original_prediction, dim=1).cpu().detach().numpy().flatten()]

        # Step 2:Run Object Detection (YOLO)
        results = yolo_model(pil_image)
        detections = results.xyxy[0]
        counterfactual_found = False
        objects_detected = "None"
        bbox_info = ""
        metrics = {}

        predicted_label_after_masking = predicted_label_before_masking  # Default to original prediction
        confidence_after_masking = confidence_before_masking  # Default confidence values

        if detections.numel() == 0:
            logging.warning(f" No objects detected in {image_filename}. Skipping masking.")
        else:
            for det in detections:
                x_min, y_min, x_max, y_max = map(int, det[:4])
                objects_detected = results.names[int(det[5])]
                bbox_info = f"({x_min}, {y_min}, {x_max}, {y_max})"

                # Step 3: Apply Object Detection-Based Masking
                masked_image = input_image.clone()
                masked_image[:, :, y_min:y_max, x_min:x_max] = 0

                # Step 4: Encode Masked Image
                latent_vector_masked = encoder(masked_image)[2]

                # Step 5: Decode Masked Latent Space
                reconstructed_masked_image = decoder(latent_vector_masked)
                reconstructed_masked_image = F.interpolate(reconstructed_masked_image, size=(input_image.shape[2], input_image.shape[3]), mode="bilinear", align_corners=False)

                # Step 6: Re-encode the Reconstructed Masked Image
                latent_vector_re_encoded = encoder(reconstructed_masked_image)[2]

                # Step 7: Classify the Re-Encoded Latent Space
                masked_prediction = classifier(latent_vector_re_encoded)
                predicted_label_after_masking_idx = torch.argmax(masked_prediction, dim=1).item()
                predicted_label_after_masking = label_mapping[predicted_label_after_masking_idx]
                confidence_after_masking = [round(float(x), 5) for x in F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()]

                # Step 8: Compare Predictions
                if predicted_label_after_masking != predicted_label_before_masking:
                    counterfactual_found = True
                    metrics = calculate_image_metrics(input_image, reconstructed_masked_image)
                    
                    # Save Images Only if Counterfactual is Found
                    save_images(
                        image_filename, input_image, masked_image, reconstructed_masked_image, IMAGE_DIRS
                    )
                break  # Stop after the first object detection mask

        # Step 9: Calculate Time Taken
        end_time = time.time()
        total_time_taken = round(end_time - start_time, 5)

        # Step 10: Update CSV
        update_results_csv(
            # "object_detection_2_class" if classifier_type == "2_class" else "object_detection_4_class",
            f"object_detection_{classifier_type}",
            image_filename, {
                "Prediction (Before Masking)": predicted_label_before_masking,
                "Confidence (Before Masking)": confidence_before_masking,
                "Prediction (After Masking)": predicted_label_after_masking,
                "Confidence (After Masking)": confidence_after_masking,
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
            }, output_csv
        )
        
        logging.info(f" Updated CSV for image {image_filename}")

if __name__ == "__main__":
    process_object_detection_masking()