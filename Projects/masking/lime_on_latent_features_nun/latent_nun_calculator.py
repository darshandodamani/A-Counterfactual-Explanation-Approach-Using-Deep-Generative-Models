# location: Projects/masking/lime_on_latent_features_nun/latent_nun_calculator.py
import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from torchvision.transforms import transforms
from PIL import Image

# ------------------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------------------
# Add Python Path for Local Modules
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "autoencoder")))

from encoder import VariationalEncoder
from classifiers_4_class import ClassifierModel as Classifier

# ------------------------------------------------------------------------------
# Model Paths and Device Configuration
# ------------------------------------------------------------------------------
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_4_classes.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class Labels Mapping
CLASS_LABELS_4_CLASS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}

# ------------------------------------------------------------------------------
# Load Models
# ------------------------------------------------------------------------------
def load_models():
    """
    Load the encoder and classifier models.
    """
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))

    encoder.eval()
    classifier.eval()

    return encoder, classifier

# Load models
encoder, classifier = load_models()

# ------------------------------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------------------------------
def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an image.
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)

# ------------------------------------------------------------------------------
# Compute NUN for Each Image
# ------------------------------------------------------------------------------
def compute_nun_for_dataset(dataset_path: str, encoder, classifier):
    """
    Computes the Nearest Unlike Neighbor (NUN) for each image in the dataset.
    
    Saves the results in a CSV file.
    """
    logging.info(f"Processing dataset: {dataset_path}")
    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])

    if not image_files:
        logging.error("No images found in dataset.")
        return
    
    all_latents = []
    all_labels = []
    image_names = []

    # Extract latent vectors for all images
    for image_filename in image_files:
        image_path = os.path.join(dataset_path, image_filename)
        image_tensor = preprocess_image(image_path)

        with torch.no_grad():
            latent_vector = encoder(image_tensor)[2].cpu().numpy().reshape(-1)
            output = classifier(torch.tensor(latent_vector, dtype=torch.float32).to(device).unsqueeze(0))
            predicted_label = torch.argmax(output, dim=1).item()

        all_latents.append(latent_vector)
        all_labels.append(predicted_label)
        image_names.append(image_filename)

    # Convert to numpy for fast computation
    all_latents_np = np.vstack(all_latents)

    # Find Nearest Unlike Neighbors
    nun_results = []
    for i, query_latent in enumerate(all_latents_np):
        query_label = all_labels[i]
        query_label_name = CLASS_LABELS_4_CLASS[query_label]
        
        # Compute distances to all other latent vectors
        distances = cdist([query_latent], all_latents_np, metric="euclidean").flatten()
        distances[i] = np.inf  # Ignore self-distance

        # Find nearest neighbor with a different label
        different_label_indices = [idx for idx, label in enumerate(all_labels) if label != query_label]
        if different_label_indices:
            nearest_nun_idx = min(different_label_indices, key=lambda idx: distances[idx])
            nun_latent = all_latents_np[nearest_nun_idx]
            nun_distance = distances[nearest_nun_idx]
            nun_label = all_labels[nearest_nun_idx]
            nun_label_name = CLASS_LABELS_4_CLASS[nun_label]
            nun_image = image_names[nearest_nun_idx]

            logging.info(f"NUN for {image_names[i]}: {nun_image}, Distance: {nun_distance}, Labels: {query_label_name} -> {nun_label_name}")
        else:
            nun_latent = np.zeros_like(query_latent)
            nun_distance = -1
            nun_label = -1
            nun_label_name = "None"
            nun_image = "None"

        # Save the result
        nun_results.append([
            image_names[i], query_label_name, nun_image, nun_label_name, nun_distance, *nun_latent
        ])

    # Define feature columns
    feature_columns = [f"latent_{i}" for i in range(all_latents_np.shape[1])]
    columns = ["Image File", "Query Label", "NUN Image", "NUN Label", "Distance"] + feature_columns

    # Save results to CSV
    results_df = pd.DataFrame(nun_results, columns=columns)
    results_csv = os.path.join("latent_vectors", "nun_values.csv")
    os.makedirs("latent_vectors", exist_ok=True)
    results_df.to_csv(results_csv, index=False)

    logging.info(f"Saved NUN results to {results_csv}")

# ------------------------------------------------------------------------------
# Run the Computation
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = "dataset/town7_dataset/test/"
    compute_nun_for_dataset(dataset_path, encoder, classifier)