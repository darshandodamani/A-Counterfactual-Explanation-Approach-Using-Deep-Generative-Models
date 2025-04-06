import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image
import os
import sys
from sklearn.metrics import silhouette_score
import json

# ---------------------- Configuration ----------------------
# Define paths
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, 'Projects/autoencoder'))

# Import Encoder & Classifier
from encoder import VariationalEncoder
from classifier import ClassifierModel

# Paths to trained models
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_4_classes.pth"

# Define class labels and colors
CLASS_NAMES = ["STOP", "GO", "RIGHT", "LEFT"]
CLASS_COLORS = ["red", "green", "yellow", "purple"]  # Explicit color mapping

# Output directory for saving plots
OUTPUT_DIR = "plots/latent_space_4_classes/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- Load Models ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained encoder
encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
encoder.eval()

# Load trained classifier for 4 classes
classifier = ClassifierModel(input_size=128, hidden_size=128, output_size=4).to(device)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))
classifier.eval()

# ---------------------- Dataset Processing ----------------------
# Define transformations
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

# Path to test dataset and labels
TEST_DIR = "dataset/town7_dataset/test/"
TEST_CSV = "dataset/town7_dataset/test/labeled_test_4_class_data_log.csv"

# Load dataset labels
df = pd.read_csv(TEST_CSV)
image_files = df["image_filename"].tolist()
labels = df["label"].map({"STOP": 0, "GO": 1, "RIGHT": 2, "LEFT": 3}).tolist()

# ---------------------- Extract Latent Features ----------------------
latent_vectors = []
true_labels = []
predicted_labels = []

# Process each test image
for img_file, label in zip(image_files, labels):
    img_path = os.path.join(TEST_DIR, img_file)
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Encode image into latent space
    with torch.no_grad():
        _, _, latent_z = encoder(image)  # Encoder outputs mu, logvar, z
        classifier_output = classifier(latent_z)  # Classifier prediction
        predicted_class = torch.argmax(classifier_output, dim=1).item()

    latent_vectors.append(latent_z.cpu().numpy().flatten())
    true_labels.append(label)
    predicted_labels.append(predicted_class)

# Convert to NumPy arrays
latent_vectors = np.array(latent_vectors)
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# ---------------------- PCA Visualization ----------------------
print("Performing PCA on Latent Features...")
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_vectors)

# Function to plot PCA and t-SNE results with explicit color mapping
def plot_latent_projection(latent_2D, labels, title, filename):
    plt.figure(figsize=(8, 6))
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_idx
        plt.scatter(latent_2D[mask, 0], latent_2D[mask, 1], label=class_name, 
                    color=CLASS_COLORS[class_idx], alpha=0.7, edgecolors='black', linewidth=0.5)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# Plot PCA (True Labels)
plot_latent_projection(latent_pca, true_labels, 
                       "PCA Projection of Latent Space (True Labels)", 
                       "pca_latent_space_true_labels.png")

# Plot PCA (Predicted Labels)
plot_latent_projection(latent_pca, predicted_labels, 
                       "PCA Projection of Latent Space (Predicted Labels)", 
                       "pca_latent_space_predicted_labels.png")

# ---------------------- t-SNE Visualization ----------------------
print("Performing t-SNE on Latent Features...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latent_tsne = tsne.fit_transform(latent_vectors)

# Plot t-SNE (True Labels)
plot_latent_projection(latent_tsne, true_labels, 
                       "t-SNE Projection of Latent Space (True Labels)", 
                       "tsne_latent_space_true_labels.png")

# Plot t-SNE (Predicted Labels)
plot_latent_projection(latent_tsne, predicted_labels, 
                       "t-SNE Projection of Latent Space (Predicted Labels)", 
                       "tsne_latent_space_predicted_labels.png")

print(f"--------- All visualizations saved in {OUTPUT_DIR} ---------")

from collections import Counter
print("True label counts:", Counter(true_labels))
print("Predicted label counts:", Counter(predicted_labels))


# ---------------------- PCA Explained Variance ----------------------
explained_variance_ratio = pca.explained_variance_ratio_
total_variance = np.sum(explained_variance_ratio[:3])  # Use top 3 for fair comparison

# ---------------------- Silhouette Score ----------------------
print("Computing Silhouette Score...")
silhouette = silhouette_score(latent_vectors, true_labels)

# ---------------------- Centroid Distance Matrix ----------------------
print("Computing Centroid Distance Matrix...")
centroids = []
for class_idx in range(len(CLASS_NAMES)):
    class_latents = latent_vectors[np.array(true_labels) == class_idx]
    centroid = np.mean(class_latents, axis=0)
    centroids.append(centroid)

# Compute Euclidean distances between class centroids
centroids = np.array(centroids)
centroid_distance_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        centroid_distance_matrix[i][j] = np.linalg.norm(centroids[i] - centroids[j])

# ---------------------- Save Metrics ----------------------
# 1. Save PCA + Silhouette Score
metrics = {
    "explained_variance_ratio": explained_variance_ratio[:3].tolist(),
    "total_variance_captured": float(total_variance),
    "silhouette_score": float(silhouette)
}

metrics_path = os.path.join(OUTPUT_DIR, "qualitative_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Qualitative metrics saved to '{metrics_path}'.")

# 2. Save centroid distance matrix
centroid_df = pd.DataFrame(centroid_distance_matrix, columns=CLASS_NAMES, index=CLASS_NAMES)
centroid_path = os.path.join(OUTPUT_DIR, "centroid_distance_matrix.csv")
centroid_df.to_csv(centroid_path)

print(f"Centroid Distance Matrix saved to '{centroid_path}'.")
