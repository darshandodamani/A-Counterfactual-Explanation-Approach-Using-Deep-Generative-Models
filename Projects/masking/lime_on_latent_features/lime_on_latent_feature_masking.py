# location: Projects/masking/lime_on_latent_features/lime_on_latent_feature_masking.py
import os
import sys
import time
import logging
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from lime.lime_tabular import LimeTabularExplainer

# Add the parent directory (where masking_utils.py is located) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

# Import utilities from masking_utils
from masking_utils import (
    load_models, METHODS_RESULTS, METHODS_HEADERS,
    calculate_image_metrics, save_images_without_mask, update_results_csv
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

LATENT_STATS_PATH = "latent_vectors/combined_median_values.csv"
PLOT_DIR_2_CLASS = "plots/lime_on_latent_feature_graphs_2_class"
PLOT_DIR_4_CLASS = "plots/lime_on_latent_feature_graphs_4_class"
os.makedirs(PLOT_DIR_2_CLASS, exist_ok=True)
os.makedirs(PLOT_DIR_4_CLASS, exist_ok=True)

# ------------------------------------------------------------------------------
# Load Latent Feature Statistics
# ------------------------------------------------------------------------------
def load_latent_statistics():
    """Load precomputed median latent values for masking strategy."""
    if os.path.exists(LATENT_STATS_PATH):
        median_values = pd.read_csv(LATENT_STATS_PATH).values.flatten()
        logging.info(f"Loaded median latent values from {LATENT_STATS_PATH}")
        return median_values
    else:
        logging.warning(f"Median latent statistics not found at {LATENT_STATS_PATH}")
        return None  # No statistics available

# median_latent_values = load_latent_statistics()
median_latent_vector = load_latent_statistics()

# ------------------------------------------------------------------------------
# Define Prediction Function for LIME
# ------------------------------------------------------------------------------
def predict_with_latent(latent_vectors: np.ndarray, classifier) -> np.ndarray:
    """Returns prediction probabilities for a batch of latent vectors."""
    latent_tensor = torch.tensor(latent_vectors, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = classifier(latent_tensor)
    return F.softmax(output, dim=1).cpu().numpy()
        
# ------------------------------------------------------------------------------
# Generate Graphs for CE Found Images
# ------------------------------------------------------------------------------
def generate_lime_feature_importance_plot(image_filename, classifier_type, lime_explanation, label_mapping):
    PLOT_DIR = PLOT_DIR_2_CLASS if classifier_type == "2_class" else PLOT_DIR_4_CLASS

    lime_explanation_map = lime_explanation.as_map()
    logging.info(f"LIME Explanation Map Keys: {list(lime_explanation_map.keys())}")

    for class_idx in [int(idx) for idx in lime_explanation_map.keys()]:  # Convert np.int64 -> int
        if class_idx not in label_mapping:
            logging.warning(f"Class index {class_idx} not found in LIME explanation map. Skipping.")
            continue

        class_label = label_mapping.get(class_idx, f"Unknown_{class_idx}")
        class_explanation = lime_explanation_map[class_idx]

        # Convert LIME weights to NumPy for easy processing
        feature_indices = np.array([idx for idx, _ in class_explanation])
        lime_weights = np.array([weight for _, weight in class_explanation], dtype=float)

        # Combine and sort by absolute weight descending
        sorted_features = sorted(zip(feature_indices, lime_weights), key=lambda x: abs(x[1]), reverse=True)
        feature_indices, lime_weights = zip(*sorted_features)

        # Color mapping
        colors = ['red' if w > 0 else 'green' for w in lime_weights]

        # Plot
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.barplot(x=list(feature_indices), y=list(lime_weights), ax=ax, palette=colors)
        ax.set_title(f"LIME Feature Importance for Class '{class_label}' - {image_filename}")
        ax.set_xlabel("Latent Feature Index (Sorted by Importance)")
        ax.set_ylabel("Feature Weight")
        ax.tick_params(axis='x', rotation=90)

        # Save Plot with High DPI
        plt.tight_layout()
        plot_filename = os.path.join(PLOT_DIR, f"{image_filename}_lime_feature_importance_class_{class_label}_sorted.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        logging.info(f" Saved sorted LIME feature importance plot for class '{class_label}' for {image_filename} at {plot_filename}")

        
# ------------------------------------------------------------------------------
# Generate Graphs for Masked Features
# ------------------------------------------------------------------------------
def generate_masked_features_plot(image_filename, classifier_type, selected_features, median_values):
    """
    Generates a plot for masked features with median replacement.
    
    Args:
        image_filename (str): Name of the image file.
        classifier_type (str): Type of classifier ("2_class" or "4_class").
        selected_features (list): List of selected feature indices.
        median_values (list): List of median values for the features.
    """
    # Select correct plot directory
    PLOT_DIR = PLOT_DIR_2_CLASS if classifier_type == "2_class" else PLOT_DIR_4_CLASS

    masked_feature_indices = selected_features
    masked_values = [median_values[idx] for idx in selected_features]

    # Masked Features Visualization
    fig, ax = plt.subplots(figsize=(18, 6))
    masking_effect = [1] * len(masked_feature_indices)
    colors_masked = ['blue' for _ in masked_feature_indices]

    # sns.barplot(x=masked_feature_indices, y=masking_effect, ax=ax, palette=colors_masked)
    sns.barplot(x=masked_feature_indices, y=masking_effect, ax=ax, hue=masked_feature_indices, palette=colors_masked, legend=False)
    for i, (idx, value) in enumerate(zip(masked_feature_indices, masked_values)):
        ax.text(i, 1.05, f"{value:.2f}", ha='center', fontsize=10)

    ax.set_title(f"Masked Features - {image_filename}")
    ax.set_xlabel("Latent Feature Index")
    ax.set_ylabel("Masking Effect")
    ax.tick_params(axis='x', rotation=90)

    # Save Plot with High DPI
    plt.tight_layout()
    plot_filename = os.path.join(PLOT_DIR, f"{image_filename}_masked_features.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    logging.info(f" Saved masked features plot for {image_filename} at {plot_filename}")

# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_lime_on_latent_masking(classifier_type: str = "4_class"):
    """
    Runs LIME on latent features for counterfactual explanation.
    
    Args:
        classifier_type (str): "2_class" or "4_class" for classification.
    """
    logging.info(f"Running LIME on Latent Feature Masking for {classifier_type} classification")

    # Step 1: Load Models
    encoder, decoder, classifier = load_models(classifier_type)
    logging.info(f"Models loaded for {classifier_type}")

    # Step 2: Select CSV File & Label Mapping
    output_csv = METHODS_RESULTS[f"lime_on_latent_feature_{classifier_type}"]
    label_mapping = CLASS_LABELS_2_CLASS if classifier_type == "2_class" else CLASS_LABELS_4_CLASS

    # Step 3: Setup Directories for Image Saving
    IMAGE_DIRS = {
        "original": f"plots/lime_on_latent_feature_{classifier_type}_original",
        "reconstructed": f"plots/lime_on_latent_feature_{classifier_type}_reconstructed"
    }
    for dir_name in IMAGE_DIRS.values():
        os.makedirs(dir_name, exist_ok=True)

    # Step 4: Initialize CSV if Not Exists
    if not os.path.exists(output_csv):
        df_results = pd.DataFrame(columns=METHODS_HEADERS[f"lime_on_latent_feature_{classifier_type}"])
        df_results.to_csv(output_csv, index=False)

    # Step 5: Iterate Through Test Images
    TEST_DIR = "dataset/town7_dataset/test/"
    for image_filename in sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".png")):
        start_time = time.time()
        image_path = os.path.join(TEST_DIR, image_filename)

        # Step 6: Load and Preprocess Image
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue

        input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

        # Save the original image
        original_image_path = os.path.join(IMAGE_DIRS["original"], image_filename)
        pil_image.save(original_image_path)
        logging.info(f"Saved original image at {original_image_path}")

        # Step 7: Extract Latent Features
        latent_vector = encoder(input_image)[2].cpu().detach().numpy().reshape(-1)

        # Step 8: Get Initial Prediction
        original_prediction = classifier(torch.tensor(latent_vector, dtype=torch.float32).to(device).unsqueeze(0))
        predicted_label_before_masking_idx = torch.argmax(original_prediction, dim=1).item()
        predicted_label_before_masking = str(label_mapping[predicted_label_before_masking_idx])
        confidence_before_masking = str([round(float(x), 5) for x in F.softmax(original_prediction, dim=1).cpu().detach().numpy().flatten()]).replace("'", "")

        # Step 9: Perform LIME Explanation
        explainer = LimeTabularExplainer(
            latent_vector.reshape(1, -1),
            mode="classification",
            feature_names=[f"latent_{i}" for i in range(len(latent_vector))],
            discretize_continuous=False
        )
        explanation = explainer.explain_instance(
            latent_vector, lambda x: predict_with_latent(x, classifier), num_features=len(latent_vector), top_labels = len(label_mapping)
        )
        
        lime_class_indices = [int(idx) for idx in explanation.as_map().keys()]
        logging.info(f"LIME Explanation Class Indices: {lime_class_indices}")

        # Step 10: Identify Important Features for Masking
        explanation_list = explanation.as_list()
        important_features = sorted(
            [(int(feature.split("_")[-1]), weight) for feature, weight in explanation_list if weight > 0],
            key=lambda x: abs(x[1]), reverse=True
        )
        lime_weights = np.array([weight for _, weight in explanation_list])

        selected_features = []
        counterfactual_found = False

        # Step 11: Apply Masking & Search for Counterfactual
        for feature_index, _ in important_features:
            masked_latent_vector = latent_vector.copy()
            masked_latent_vector[feature_index] = median_latent_vector[feature_index]
            selected_features.append(feature_index)

            masked_latent_tensor = torch.tensor(masked_latent_vector, dtype=torch.float32).to(device).unsqueeze(0)
            reconstructed_image = decoder(masked_latent_tensor)
            reconstructed_image = F.interpolate(reconstructed_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)

            re_encoded_latent = encoder(reconstructed_image)[2]
            masked_prediction = classifier(re_encoded_latent)

            predicted_label_after_masking_idx = torch.argmax(masked_prediction, dim=1).item()
            predicted_label_after_masking = str(label_mapping[predicted_label_after_masking_idx])
            confidence_after_masking = str([round(float(x), 5) for x in F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()]).replace("'", "")

            if predicted_label_after_masking_idx != predicted_label_before_masking_idx:
                counterfactual_found = True
                sparsity = round(np.sum(latent_vector != masked_latent_vector), 5)
                proximity = round(np.linalg.norm(latent_vector - masked_latent_vector), 5)

                
                metrics = calculate_image_metrics(input_image, reconstructed_image)
                generate_lime_feature_importance_plot(image_filename, classifier_type, explanation, label_mapping)
                generate_masked_features_plot(image_filename, classifier_type, selected_features, median_latent_vector)
                
                # Save the reconstructed image
                reconstructed_image_path = os.path.join(IMAGE_DIRS["reconstructed"], image_filename)
                reconstructed_image_pil = transforms.ToPILImage()(reconstructed_image.squeeze().cpu())
                reconstructed_image_pil.save(reconstructed_image_path)
                logging.info(f"Saved reconstructed image at {reconstructed_image_path}")

                break # Stop after finding the first counterfactual

        # Step 12: Calculate Processing Time
        end_time = time.time()
        total_time_taken = round(end_time - start_time, 5)

        # Step 13: **Fix Formatting for CSV**
        if selected_features:
            feature_selection_percentage = f"{round((len(selected_features) / len(latent_vector)) * 100, 2)}%"
            selected_features_str = str(selected_features)
        else:
            feature_selection_percentage = ""
            selected_features_str = ""


        # Step 14: Update CSV with Results
        update_results_csv(
            f"lime_on_latent_feature_{classifier_type}",
            image_filename,
            {
                "Prediction (Before Masking)": predicted_label_before_masking,
                "Confidence (Before Masking)": confidence_before_masking,
                "Prediction (After Masking)": predicted_label_after_masking,
                "Confidence (After Masking)": confidence_after_masking,
                "Counterfactual Found": counterfactual_found,
                "Feature Selection (%)": feature_selection_percentage,
                "Selected Features": selected_features_str,
                "Sparsity": sparsity if counterfactual_found else "",
                "Proximity": proximity if counterfactual_found else "",
                "SSIM": metrics.get("SSIM", "") if counterfactual_found else "",
                "MSE": metrics.get("MSE", "") if counterfactual_found else "",
                "PSNR": metrics.get("PSNR", "") if counterfactual_found else "",
                "UQI": metrics.get("UQI", "") if counterfactual_found else "",
                "VIFP": metrics.get("VIFP", "") if counterfactual_found else "",
                "Time Taken (s)": total_time_taken
            },
            output_csv,
        )

if __name__ == "__main__":
    process_lime_on_latent_masking()