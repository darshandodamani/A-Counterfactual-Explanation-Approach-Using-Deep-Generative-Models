# location; Projects/masking/masking_utils.py
import os
import logging
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import multiprocessing
from typing import Dict, Tuple
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import sys
from typing import Any, Dict
from PIL import ImageDraw

# spawn is used for multiprocessing to avoid CUDA reinitialization errors
multiprocessing.set_start_method("spawn", force=True)

# Autoencoder directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "autoencoder")))

# ----------------------------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ----------------------------------------------------------------------------
# Dictionary mapping method names to their results CSV file paths
# ----------------------------------------------------------------------------
METHODS_RESULTS: Dict[str, str] = {
    "grid_based_2_class": "results/masking/grid_based/grid_based_masking_2_classes_results.csv",
    "grid_based_4_class": "results/masking/grid_based/grid_based_masking_4_classes_results.csv",
    "lime_on_image_2_class": "results/masking/lime_on_images/lime_on_image_masking_2_classes_results.csv",
    "lime_on_image_4_class": "results/masking/lime_on_images/lime_on_image_masking_4_classes_results.csv",
    "object_detection_2_class": "results/masking/object_detection/object_detection_masking_2_classes_results.csv",
    "object_detection_4_class": "results/masking/object_detection/object_detection_masking_4_classes_results.csv", 
    "lime_on_latent_feature_2_class": "results/masking/lime_on_latent/lime_on_latent_masking_2_classes_results.csv",
    "lime_on_latent_feature_4_class": "results/masking/lime_on_latent/lime_on_latent_masking_4_classes_results.csv",
     "lime_on_latent_feature_2_class_NUN": "results/masking/lime_on_latent/2_class_NUN_results.csv",
    "lime_on_latent_feature_4_class_NUN": "results/masking/lime_on_latent/4_class_NUN_results.csv",
    "shap_on_latent_feature_2_class": "results/masking/lime_on_latent/shap_on_latent_masking_2_classes_results.csv",
    "shap_on_latent_feature_4_class": "results/masking/lime_on_latent/shap_on_latent_masking_4_classes_results.csv"
}

# ----------------------------------------------------------------------------
# Column headers for each method
# ----------------------------------------------------------------------------
METHODS_HEADERS: Dict[str, list] = {
    "lime_on_latent_feature_2_class_NUN": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", 
        "Counterfactual Found", "Features Replaced", "Feature Selection (%)",
        "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
    ],
    "lime_on_latent_feature_4_class_NUN": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", 
        "Counterfactual Found", "Features Replaced", "Feature Selection (%)",
        "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
    ],
    "grid_based_2_class": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
    ],
    "grid_based_4_class": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
    ],
    "lime_on_image_2_class": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
    ],
    "lime_on_image_4_class": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
    ],
    "object_detection_2_class": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Objects Detected",
        "Time Taken (s)"
    ],
    "object_detection_4_class": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Objects Detected",
        "Time Taken (s)"
    ],
    "lime_on_latent_feature_2_class": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Feature Selection (%)", "Selected Features", "SSIM", "MSE", "PSNR", "UQI", "VIFP",
        "Time Taken (s)"
    ],
    "lime_on_latent_feature_4_class": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Feature Selection (%)", "Selected Features", "SSIM", "MSE", "PSNR", "UQI", "VIFP",
        "Time Taken (s)"
    ],
    "shap_on_latent_feature_2_class": [
        "Image File", "Prediction (Before Masking)", "Prediction (After Masking)", "Counterfactual Found",
        "Selected Features", "Time Taken (s)"
    ],
    "shap_on_latent_feature_4_class": [
        "Image File", "Prediction (Before Masking)", "Counterfactual Found",
        "Selected Features", "Time Taken (s)"
    ]
}

# ----------------------------------------------------------------------------
# Utility Functions for Masking Methods
# ----------------------------------------------------------------------------
def load_models(classifier_type="4_class") -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """
    Loads encoder, decoder, and classifier models based on the specified classifier type.
    """
    ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
    DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"
    CLASSIFIER_PATH = (
        "model/epochs_500_latent_128_town_7/classifier_4_classes.pth"
        if classifier_type == "4_class"
        else "model/epochs_500_latent_128_town_7/classifier_final.pth"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from encoder import VariationalEncoder
    from decoder import Decoder
    if classifier_type == "4_class":
        from classifiers_4_class import ClassifierModel as Classifier
    else:
        from classifier import ClassifierModel as Classifier

    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    classifier.eval()

    logging.info(f"Models loaded for {classifier_type} classification.")
    return encoder, decoder, classifier



def initialize_method_results(method_name: str, results_csv: str, headers: list):
    """
    Ensures that the specified method has a results CSV file with the correct headers.
    """
    if not os.path.exists(results_csv):
        df = pd.DataFrame(columns=headers)
        df.to_csv(results_csv, index=False)
        logging.info(f"Initialized {results_csv} with headers.")
    else:
        logging.info(f"Results CSV for {method_name} already exists. Skipping initialization.")


def calculate_image_metrics(original: torch.Tensor, modified: torch.Tensor) -> Dict[str, float]:
    """
    Computes image quality metrics including SSIM, MSE, PSNR, UQI, and VIFP.
    """
    original_np = original.cpu().squeeze().numpy().transpose(1, 2, 0)
    modified_np = modified.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
    
    original_np = (original_np * 255).astype(np.uint8)
    modified_np = (modified_np * 255).astype(np.uint8)
    
    return {
        "SSIM": round(ssim(original_np, modified_np, channel_axis=-1, data_range=255), 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np, data_range=255), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }


def save_images(image_filename: str, input_image: torch.Tensor, masked_image: torch.Tensor, 
                 reconstructed_image: torch.Tensor, output_dirs: Dict[str, str]) -> None:
    """
    Saves images (original, masked, and reconstructed) to the specified directories.
    """
    os.makedirs(output_dirs["original"], exist_ok=True)
    os.makedirs(output_dirs["masked"], exist_ok=True)
    os.makedirs(output_dirs["reconstructed"], exist_ok=True)
    
    original_pil = to_pil_image(input_image.squeeze(0).cpu())
    masked_pil = to_pil_image(masked_image.squeeze(0).cpu())
    reconstructed_pil = to_pil_image(reconstructed_image.squeeze(0).cpu())
    
    base_filename, _ = os.path.splitext(image_filename)
    
    original_pil.save(os.path.join(output_dirs["original"], f"{base_filename}.png"))
    masked_pil.save(os.path.join(output_dirs["masked"], f"{base_filename}.png"))
    reconstructed_pil.save(os.path.join(output_dirs["reconstructed"], f"{base_filename}.png"))
    
    logging.info(f"Saved images for {image_filename}")
    
from PIL import ImageDraw

def save_images_with_bounding_box(image_filename: str, input_image: torch.Tensor, masked_image: torch.Tensor, 
                                  reconstructed_image: torch.Tensor, output_dirs: Dict[str, str], bbox_info: str = None) -> None:
    """
    Saves images (original, masked, reconstructed, and original reconstructed) to the specified directories.
    Also draws a bounding box on the original image if bbox_info is provided.
    """
    # Create directories
    os.makedirs(output_dirs["original"], exist_ok=True)
    os.makedirs(output_dirs["masked"], exist_ok=True)
    os.makedirs(output_dirs["reconstructed"], exist_ok=True)
    os.makedirs(output_dirs.get("original_with_bbox", f"{output_dirs['original']}_with_bbox"), exist_ok=True)
    
    original_pil = to_pil_image(input_image.squeeze(0).cpu())
    masked_pil = to_pil_image(masked_image.squeeze(0).cpu())
    reconstructed_pil = to_pil_image(reconstructed_image.squeeze(0).cpu())
    
    base_filename, _ = os.path.splitext(image_filename)
    
    # Save original, masked, and reconstructed images
    original_pil.save(os.path.join(output_dirs["original"], f"{base_filename}.png"))
    masked_pil.save(os.path.join(output_dirs["masked"], f"{base_filename}.png"))
    reconstructed_pil.save(os.path.join(output_dirs["reconstructed"], f"{base_filename}.png"))
    
    # Save the original reconstructed image (without masking)
    original_reconstructed_pil = to_pil_image(reconstructed_image.squeeze(0).cpu())
    os.makedirs(output_dirs.get("original_reconstructed", f"{output_dirs['original']}_reconstructed"), exist_ok=True)
    original_reconstructed_pil.save(os.path.join(output_dirs.get("original_reconstructed", f"{output_dirs['original']}_reconstructed"), f"{base_filename}_original_reconstructed.png"))
    
    # Draw bounding box on original image if bbox_info is provided
    if bbox_info:
        draw = ImageDraw.Draw(original_pil)
        x_min, y_min, x_max, y_max = map(int, eval(bbox_info))
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        original_pil.save(os.path.join(output_dirs.get("original_with_bbox", f"{output_dirs['original']}_with_bbox"), f"{base_filename}_with_bbox.png"))
    
    logging.info(f"Saved images for {image_filename}")
    
def save_images_without_mask(image_filename: str, input_image: torch.Tensor, reconstructed_image: torch.Tensor, output_dirs: Dict[str, str]) -> None:
    """
    Saves only the original and reconstructed images (without masked image).
    Specifically for LIME on latent feature masking, where no masked image exists.
    """
    os.makedirs(output_dirs["original"], exist_ok=True)
    os.makedirs(output_dirs["reconstructed"], exist_ok=True)

    original_pil = to_pil_image(input_image.squeeze(0).cpu())
    reconstructed_pil = to_pil_image(reconstructed_image.squeeze(0).cpu())

    base_filename, _ = os.path.splitext(image_filename)

    original_pil.save(os.path.join(output_dirs["original"], f"{base_filename}.png"))
    reconstructed_pil.save(os.path.join(output_dirs["reconstructed"], f"{base_filename}.png"))

    logging.info(f" Saved original & reconstructed images for {image_filename} (No masked image)")

def update_results_csv(method: str, image_filename: str, results: Dict[str, Any], results_csv: str):
    logging.info(f" Loading {results_csv} for update...")
    
    if not os.path.exists(results_csv):
        logging.info(f"{results_csv} not found. Creating new CSV with headers.")
        df_results = pd.DataFrame(columns=METHODS_HEADERS[method])
        df_results.to_csv(results_csv, index=False)

    df_results = pd.read_csv(results_csv)

    # To ensure proper type casting before updating DataFrame
    formatted_results = {}
    for key, value in results.items():
        if key == "Image File":
            formatted_results[key] = image_filename
        elif isinstance(value, bool):  
            formatted_results[key] = bool(value)
        elif isinstance(value, (int, float, np.float64, np.int64)):  
            formatted_results[key] = float(value)
        elif isinstance(value, (list, np.ndarray)):  
            formatted_results[key] = ", ".join(map(str, value))
        elif value == "":  
            formatted_results[key] = np.nan
        else:
            formatted_results[key] = str(value)

    logging.info(f"Formatted results before update: {formatted_results}")

    # Ensure correct dtype for each column before updating
    for col in df_results.columns:
        if col in formatted_results:
            if df_results[col].dtype == "bool":
                formatted_results[col] = bool(formatted_results[col])
            elif df_results[col].dtype == "float64":
                try:
                    formatted_results[col] = float(formatted_results[col])
                except ValueError:
                    logging.warning(f"ValueError: Could not convert {formatted_results[col]} to float. Setting as NaN.")
                    formatted_results[col] = np.nan

    # Update CSV File
    if image_filename in df_results["Image File"].values:
        logging.info(f"Updating existing row for {image_filename}")
        df_results.loc[df_results["Image File"] == image_filename, list(formatted_results.keys())] = list(formatted_results.values())
    else:
        logging.info(f"Appending new row for {image_filename}")
        new_row = {"Image File": image_filename, **formatted_results}
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)

    # Ensure proper datatype for float columns
    float_columns = ["SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"]
    for col in float_columns:
        if col in df_results.columns:
            df_results[col] = pd.to_numeric(df_results[col], errors="coerce") 

    logging.info(f" Data to be written:\n{df_results.tail()}")

    # Ensure CSV is written properly
    df_results.to_csv(results_csv, index=False)
    
    logging.info(f" Successfully updated {results_csv} with results for {image_filename}")


def process_grid_based_masking():
    """
    Calls the grid-based masking script and executes the process.
    """
    from grid_based_masking.grid_based_masking import process_grid_based_masking
    classifer_types = ["2_class", "4_class"]
    
    processes = []
    for classifier_type in classifer_types:
        logging.info(f"Starting grid-based masking for {classifier_type} in parallel.")
        p = multiprocessing.Process(target=process_grid_based_masking, args=(classifier_type,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        logging.info("Completed grid-based masking.")
    
def process_object_detection_based_masking():
    """
    Calls the grid-based masking script and executes the process.
    """
    from object_detection.object_detection_based_masking import process_object_detection_masking
    classifer_types = ["2_class", "4_class"]
    
    processes = []
    for classifier_type in classifer_types:
        logging.info(f"Starting object detection masking for {classifier_type} in parallel.")
        p = multiprocessing.Process(target=process_object_detection_masking, args=(classifier_type,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        logging.info("Completed object detection masking.")
    
def process_lime_on_image_masking():
    """
    Calls the LIME on Image masking script and executes the process.
    """
    from lime_on_images.lime_on_image_masking import process_lime_on_image_masking
    classifer_types = ["2_class", "4_class"]
    
    processes = []
    for classifier_type in classifer_types:
        logging.info(f"Starting LIME on Image masking for {classifier_type} in parallel.")
        p = multiprocessing.Process(target=process_lime_on_image_masking, args=(classifier_type,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        logging.info("Completed LIME on Image masking.")
    
def process_lime_on_latent_masking():
    """
    Calls the LIME on Latent masking script and executes the process.
    """
    from lime_on_latent_features.lime_on_latent_feature_masking import process_lime_on_latent_masking
    classifer_types = ["2_class", "4_class"]
    
    processes = []
    for classifier_type in classifer_types:
        logging.info(f"Starting LIME on Latent masking for {classifier_type} in parallel.")
        p = multiprocessing.Process(target=process_lime_on_latent_masking, args=(classifier_type,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        logging.info("Completed LIME on Latent masking.")
        
def process_lime_on_latent_nun_masking():
    """
    Calls the LIME on Latent masking script using NUN and executes the process.
    """
    from lime_on_latent_features.lime_on_latent_feature_masking_using_NUN import process_lime_on_latent_masking_nun
    classifer_types = ["2_class", "4_class"]
    
    processes = []
    for classifier_type in classifer_types:
        logging.info(f"Starting LIME on Latent masking using NUN for {classifier_type} in parallel.")
        p = multiprocessing.Process(target=process_lime_on_latent_masking_nun, args=(classifier_type,))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        logging.info(f"Completed LIME on Latent masking using NUN for {classifier_type}.")
        
def run_parallel_masking():
    """
    Runs all masking methods in parallel using multiprocessing.
    """
    methods = {
        "grid_based": process_grid_based_masking,
        "object_detection": process_object_detection_based_masking,
        "lime_on_image": process_lime_on_image_masking,
        "lime_on_latent": process_lime_on_latent_masking,
        "lime_on_latent_nun": process_lime_on_latent_nun_masking,
    }
    
    processes = []
    for method_name, method_func in methods.items():
        p = multiprocessing.Process(target=method_func)
        processes.append(p)
        p.start()
        logging.info(f"Started {method_name} masking in parallel.")
    
    for p in processes:
        p.join()
        logging.info(f"Completed {method_name} method.")


if __name__ == "__main__":
    logging.info("Initializing models and starting parallel masking...")
    run_parallel_masking()
    logging.info("All masking methods completed execution.")
