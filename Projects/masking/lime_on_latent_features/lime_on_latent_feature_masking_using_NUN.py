# location: Projects/masking/lime_on_latent_features/lime_on_latent_feature_masking_using_NUN.py
import os
import sys
import time
import logging
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from lime.lime_tabular import LimeTabularExplainer
from scipy.spatial.distance import cdist
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory for utility imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

# Import utilities from masking_utils
from masking_utils import (
    load_models, METHODS_RESULTS, METHODS_HEADERS,
    calculate_image_metrics, update_results_csv
)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants and Paths
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS_4_CLASS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}
CLASS_LABELS_2_CLASS = {0: "STOP", 1: "GO"}
TEST_DIR = "dataset/town7_dataset/test/"

# Define Prediction Function
def predict_with_latent(latent_vectors: np.ndarray, classifier) -> np.ndarray:
    latent_tensor = torch.tensor(latent_vectors, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = classifier(latent_tensor)
    return F.softmax(output, dim=1).cpu().numpy()

# Find Nearest Unlike Neighbor (NUN)
def find_nearest_unlike_neighbor(query_latent, query_label, encoder, classifier, dataset_path, classifier_type):
    NUN_DIR = f"plots/lime_with_nun_{classifier_type}/nun_images"
    os.makedirs(NUN_DIR, exist_ok=True)
    
    all_latent_vectors, all_labels, all_filenames = [], [], []
    for image_filename in sorted(os.listdir(dataset_path)):
        if not image_filename.endswith(".png"):
            continue
        image_path = os.path.join(dataset_path, image_filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        latent_vector = encoder(input_tensor)[2].cpu().detach().numpy().reshape(-1)
        output = classifier(torch.tensor(latent_vector, dtype=torch.float32).to(device).unsqueeze(0))
        predicted_label = torch.argmax(output, dim=1).item()
        if predicted_label != query_label:
            all_latent_vectors.append(latent_vector)
            all_labels.append(predicted_label)
            all_filenames.append(image_filename)
    if not all_latent_vectors:
        return query_latent, None, None  # Return query if no unlike neighbors exist
    distances = cdist([query_latent], all_latent_vectors, metric="euclidean").flatten()
    min_index = np.argmin(distances)
    nun_latent = all_latent_vectors[min_index]
    nun_filename = all_filenames[min_index]
    
    # Save NUN image
    nun_image = Image.open(os.path.join(dataset_path, nun_filename)).convert("RGB")
    nun_image.save(os.path.join(NUN_DIR, f"nun_{nun_filename}"))
    
    return nun_latent, nun_filename, nun_image

def plot_side_by_side_images(query_image, nun_image, reconstructed_image, query_filename, nun_filename, classifier_type):
    SAVE_DIR = f"plots/lime_with_nun_{classifier_type}/side_by_side_images"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original/query image
    ax1.imshow(query_image)
    ax1.set_title(f"Original/Query Image\n{query_filename}")
    ax1.axis('off')
    
    # Plot NUN image
    ax2.imshow(nun_image)
    ax2.set_title(f"NUN Image\n{nun_filename}")
    ax2.axis('off')
    
    # Plot reconstructed image
    ax3.imshow(reconstructed_image)
    ax3.set_title("Reconstructed Image")
    ax3.axis('off')
    
    plt.tight_layout()
    
    plot_filename = os.path.join(SAVE_DIR, f"{query_filename}_side_by_side.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    logging.info(f"Saved side-by-side image plot at {plot_filename}")

def plot_lime_explanation(explanation, image_filename, classifier_type):
    SAVE_DIR = f"plots/lime_with_nun_{classifier_type}/lime_explanations"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    explanation_list = explanation.as_list()
    explanation_list_sorted = sorted(explanation_list, key=lambda x: abs(x[1]), reverse=True)
    features = [feat for feat, wt in explanation_list_sorted]
    weights = [wt for feat, wt in explanation_list_sorted]
    
    # Create a color mapping for positive and negative weights
    colors = ['red' if wt > 0 else 'green' for wt in weights]
    
    plt.figure(figsize=(24, 12))
    # Use hue instead of palette, and set legend to False
    sns.barplot(x=features, y=weights, hue=features, palette=colors, legend=False)
    plt.title(f"LIME Explanation for {image_filename}")
    plt.xlabel("Latent Feature")
    plt.ylabel("Feature Weight")
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plot_filename = os.path.join(SAVE_DIR, f"{image_filename}_lime_explanation.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    logging.info(f"Saved LIME explanation plot at {plot_filename}")

def plot_replaced_features(query_latent, nun_latent, replaced_features, image_filename, classifier_type):
    SAVE_DIR = f"plots/lime_with_nun_{classifier_type}/replaced_features"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    plt.figure(figsize=(24, 12))
    plt.plot(query_latent, label='Query Latent', alpha=0.5)
    plt.plot(nun_latent, label='NUN Latent', alpha=0.5)
    
    for feature in replaced_features:
        plt.scatter(feature, query_latent[feature], color='red', s=50, zorder=5)
        plt.scatter(feature, nun_latent[feature], color='green', s=50, zorder=5)
        plt.plot([feature, feature], [query_latent[feature], nun_latent[feature]], color='black', linestyle='--', zorder=4)
    
    plt.title(f"Replaced Features for {image_filename}")
    plt.xlabel("Latent Feature Index")
    plt.ylabel("Latent Feature Value")
    plt.legend()
    plt.tight_layout()
    
    plot_filename = os.path.join(SAVE_DIR, f"{image_filename}_replaced_features.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    logging.info(f"Saved replaced features plot at {plot_filename}")

# Process LIME on Latent Masking Using NUN (No Restriction)
def process_lime_on_latent_masking_nun(classifier_type: str = "4_class"):
    encoder, decoder, classifier = load_models(classifier_type)
    output_csv = METHODS_RESULTS[f"lime_on_latent_feature_{classifier_type}_NUN"]
    
    if not os.path.exists(output_csv):
        # Create an empty DataFrame with the correct headers
        df_results = pd.DataFrame(columns=METHODS_HEADERS[f"lime_on_latent_feature_{classifier_type}_NUN"])
        df_results.to_csv(output_csv, index=False)
        logging.info(f"Initialized {output_csv} with headers.")

    label_mapping = CLASS_LABELS_4_CLASS if classifier_type == "4_class" else CLASS_LABELS_2_CLASS

    for image_filename in sorted(os.listdir(TEST_DIR)):
        if not image_filename.endswith(".png"):
            continue

        start_time = time.time()
        image_path = os.path.join(TEST_DIR, image_filename)
        pil_image = Image.open(image_path).convert("RGB")
        input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

        latent_vector = encoder(input_image)[2].cpu().detach().numpy().reshape(-1)
        total_features = len(latent_vector)  # Total number of latent features

        original_prediction = classifier(torch.tensor(latent_vector, dtype=torch.float32).to(device).unsqueeze(0))
        predicted_label_before_masking_idx = torch.argmax(original_prediction, dim=1).item()
        predicted_label_before_masking = label_mapping.get(predicted_label_before_masking_idx, "Unknown")
        confidence_before_masking = str([round(float(x), 5) for x in F.softmax(original_prediction, dim=1).cpu().detach().numpy().flatten()]).replace("'", "")
        logging.info(f"Image: {image_filename} | Prediction Before Masking: {predicted_label_before_masking}")

        nun_latent_vector, nun_filename, nun_image = find_nearest_unlike_neighbor(latent_vector, predicted_label_before_masking_idx, encoder, classifier, TEST_DIR, classifier_type)

        explainer = LimeTabularExplainer(
            nun_latent_vector.reshape(1, -1),
            mode="classification",
            feature_names=[f"latent_{i}" for i in range(len(nun_latent_vector))],
            discretize_continuous=False
        )
        explanation = explainer.explain_instance(
            nun_latent_vector,
            lambda x: predict_with_latent(x, classifier),
            num_features=len(nun_latent_vector),
            top_labels=len(label_mapping)
        )

        nun_feature_order = [int(feature.split("_")[-1]) for feature, weight in explanation.as_list()]

        masked_latent_vector = latent_vector.copy()
        replaced_features = []
        counterfactual_found = False

        for feature_index in nun_feature_order:  # No restriction, modify all necessary features
            masked_latent_vector[feature_index] = nun_latent_vector[feature_index]
            replaced_features.append(feature_index)

            num_features_used = len(replaced_features)
            percentage_changed = (num_features_used / total_features) * 100 if total_features > 0 else 0

            masked_latent_tensor = torch.tensor(masked_latent_vector, dtype=torch.float32).to(device).unsqueeze(0)
            reconstructed_image = decoder(masked_latent_tensor)
            reconstructed_image = F.interpolate(reconstructed_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)
            re_encoded_latent = encoder(reconstructed_image)[2]
            masked_prediction = classifier(re_encoded_latent)
            predicted_label_after_masking_idx = torch.argmax(masked_prediction, dim=1).item()
            predicted_label_after_masking = label_mapping.get(predicted_label_after_masking_idx, "Unknown")
            confidence_after_masking = str([round(float(x), 5) for x in F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()]).replace("'", "")
            
            logging.info(f"Image: {image_filename} | Features Used: {num_features_used} | Feature Selection: {percentage_changed:.2f}%")
            logging.info(f"Image: {image_filename} | Prediction After Masking: {predicted_label_after_masking}")

            if predicted_label_after_masking_idx != predicted_label_before_masking_idx:
                counterfactual_found = True
                break  # Stop replacing features once we get a counterfactual

        total_time_taken = round(time.time() - start_time, 5)
        metrics = calculate_image_metrics(input_image, reconstructed_image)

        # Convert tensors to PIL images for plotting
        query_image_pil = transforms.ToPILImage()(input_image.squeeze().cpu())
        reconstructed_image_pil = transforms.ToPILImage()(reconstructed_image.squeeze().cpu())

        # Plot side-by-side images
        plot_side_by_side_images(query_image_pil, nun_image, reconstructed_image_pil, image_filename, nun_filename, classifier_type)
        
        plot_lime_explanation(explanation, nun_filename, classifier_type=classifier_type)
        plot_replaced_features(latent_vector, nun_latent_vector, replaced_features, image_filename, classifier_type)

        update_results_csv(
            f"lime_on_latent_feature_{classifier_type}_NUN",
            image_filename,
            {
                "Prediction (Before Masking)": predicted_label_before_masking,
                "Confidence (Before Masking)": confidence_before_masking,
                "Prediction (After Masking)": predicted_label_after_masking,
                "Confidence (After Masking)": confidence_after_masking,              
                "Counterfactual Found": counterfactual_found,
                "Features Replaced": num_features_used,  # Features used before flip
                "Feature Selection (%)": f"{percentage_changed:.2f}%",  # Percentage of modified features
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
    process_lime_on_latent_masking_nun()

