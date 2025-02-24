import os
import sys
import time
import logging
import warnings
from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi

# Set the TORCH_HOME environment variable to the desired directory
os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), 'model')

# ------------------------------------------------------------------------------
# Suppress specific FutureWarning from torch.utils.checkpoint and others
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.")
warnings.filterwarnings("ignore", message=".*torch\\.cuda\\.amp\\.autocast.*", category=FutureWarning)

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
OUTPUT_CSV = "results/masking/object_detection_masking_4_classes_results.csv"
DEBUG_CSV = "results/masking/object_detection_masking_debug_4_classes.csv"
CE_LOG_CSV = "results/masking/object_detection_masking_ce_logs_4_classes.csv"
TEST_DIR = "dataset/town7_dataset/test/"

# Directories for Saving Images
IMAGE_DIRS = {
    "original": "plots/object_detection_original",
    "masked": "plots/object_detection_masked",
    "reconstructed": "plots/object_detection_reconstructed",
    "difference": "plots/object_detection_difference"
}

for dir_path in IMAGE_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def save_image(tensor, path):
    """Converts tensor to PIL image and saves it."""
    img = to_pil_image(tensor.squeeze(0).cpu())
    img.save(path)

def calculate_image_metrics(original, modified):
    """Computes similarity metrics between original and modified images."""
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

# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_object_detection_masking():
    """
    Runs object detection-based masking on each test image, updates the results CSV,
    and saves individual images at various stages.
    """
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(OUTPUT_CSV)
    
    # Load models
    encoder, decoder, classifier = load_models()
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, autoshape=True)

    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(TEST_DIR, image_filename)
        
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue
        
        input_image = transform(pil_image).unsqueeze(0).to(device)
        original_prediction = row["Prediction (Before Masking)"]

        counterfactual_found = False
        final_prediction = original_prediction
        objects_detected = "None"
        confidence_final = None
        bbox_info = "None"
        metrics = {}

        with torch.no_grad():
            results = yolo_model(pil_image)
            detections = results.xyxy[0]  # Bounding boxes: [x_min, y_min, x_max, y_max, confidence, class]

            if detections.numel() == 0:  # No objects detected
                logging.warning(f"⚠️ No objects detected in {image_filename}. Skipping masking.")
                confidence_final = "N/A"  # Ensure this is set to avoid NoneType error
            else:
                for det in detections:
                    x_min, y_min, x_max, y_max = map(int, det[:4])
                    objects_detected = results.names[int(det[5])]
                    bbox_info = f"({x_min}, {y_min}, {x_max}, {y_max})"

                    # Step 1: Apply Object Detection-Based Masking
                    masked_image = input_image.clone()
                    masked_image[:, :, y_min:y_max, x_min:x_max] = 0

                    # Step 2: Encode Masked Image
                    latent_vector_masked = encoder(masked_image)[2]

                    # Step 3: Decode Masked Latent Space
                    reconstructed_masked_image = decoder(latent_vector_masked)

                    # Ensure reconstructed masked image is resized before re-encoding
                    if reconstructed_masked_image.shape[1:] != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
                        reconstructed_masked_image = F.interpolate(reconstructed_masked_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)

                    # Flatten the image before sending it to the encoder
                    reconstructed_masked_image = reconstructed_masked_image.view(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

                    # Step 4: Re-encode the Reconstructed Masked Image
                    latent_vector_re_encoded = encoder(reconstructed_masked_image)[2]  # Ensure correct shape


                    # Step 5: Classify the Re-Encoded Latent Space
                    masked_prediction = classifier(latent_vector_re_encoded)

                    confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
                    predicted_label_after_masking = CLASS_LABELS[torch.argmax(masked_prediction, dim=1).item()]
                    
                    # Step 6: Compare Predictions
                    if predicted_label_after_masking != original_prediction:
                        counterfactual_found = True
                        final_prediction = predicted_label_after_masking
                        metrics = calculate_image_metrics(input_image, reconstructed_masked_image)

                        break  # Stop after first counterfactual

        # **✅ Correct Time Calculation**
        end_time = time.time()
        previous_time = float(row.get("Time Taken (s)", 0))  # Get previous processing time if exists
        total_time_taken = round(end_time - start_time + previous_time, 5)

        # Format confidence_final exactly like in the 2-class version:
        if confidence_final is not None and isinstance(confidence_final, (list, np.ndarray)):
            # Convert each element to float to ensure consistent formatting, then format to five decimals
            confidence_final_str = "[" + ", ".join(f"{float(x):.5f}" for x in confidence_final) + "]"
        else:
            confidence_final_str = ""  # Leave empty if no counterfactual is found



        # Ensure bounding box values are stored in the same format as Grid Size & Grid Position
        bbox_dimensions = bbox_info if bbox_info != "None" else ""  # Empty instead of "N/A"
        bbox_position = bbox_info if bbox_info != "None" else ""  # Empty instead of "N/A"

        # **✅ Save Images Only When Counterfactual is Found (CE=True)**
        if counterfactual_found:
            save_image(masked_image, os.path.join(IMAGE_DIRS["masked"], image_filename))
            save_image(reconstructed_masked_image, os.path.join(IMAGE_DIRS["reconstructed"], image_filename))
            save_image((input_image - reconstructed_masked_image).abs(), os.path.join(IMAGE_DIRS["difference"], image_filename))

        # Update the results DataFrame for the current image
        df_results.loc[df_results["Image File"] == image_filename, [
            "Prediction (After Masking)",
            "Confidence (After Masking)",
            "Counterfactual Found",
            "Grid Size",        # Storing Bounding Box dimensions
            "Grid Position",    # Storing Bounding Box position
            "SSIM",
            "MSE",
            "PSNR",
            "UQI",
            "VIFP",
            "Objects Detected",
            "Time Taken (s)"
        ]] = [
            str(final_prediction),  
            confidence_final_str,  
            str(counterfactual_found),  
            bbox_dimensions,   # Store as Grid Size (for consistency) or leave empty
            bbox_position,     # Store as Grid Position (for consistency) or leave empty
            str(metrics.get("SSIM", "")),  # Empty if not available
            str(metrics.get("MSE", "")),   # Empty if not available
            str(metrics.get("PSNR", "")),  # Empty if not available
            str(metrics.get("UQI", "")),   # Empty if not available
            str(metrics.get("VIFP", "")),  # Empty if not available
            objects_detected,  # Properly formatted detected objects
            str(total_time_taken)
        ]

        df_results.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Updated CSV for image {image_filename}")


if __name__ == "__main__":
    process_object_detection_masking()