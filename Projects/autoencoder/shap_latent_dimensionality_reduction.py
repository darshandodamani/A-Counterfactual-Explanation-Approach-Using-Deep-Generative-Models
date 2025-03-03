import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Import VAE components
from encoder import VariationalEncoder
from decoder import Decoder

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Load Pretrained Encoder and Decoder
# ------------------------------------------------------------------------------
LATENT_DIM = 128  # Original latent space size
NEW_LATENT_DIM = 40  # Target reduced latent space size

ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"

encoder = VariationalEncoder(latent_dims=LATENT_DIM, num_epochs=100).to(device)
decoder = Decoder(latent_dims=LATENT_DIM, num_epochs=100).to(device)

encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device, weights_only=True))

encoder.eval()
decoder.eval()
print(f" Models Loaded: Encoder and Decoder (Latent Dim: {LATENT_DIM})")

# ------------------------------------------------------------------------------
# Load a Sample Image for SHAP Analysis
# ------------------------------------------------------------------------------
IMAGE_PATH = "dataset/town7_dataset/test/town7_000003.png"
pil_image = Image.open(IMAGE_PATH).convert("RGB")
transform = transforms.ToTensor()
input_image = transform(pil_image).unsqueeze(0).to(device)

# ------------------------------------------------------------------------------
# Extract Latent Representation
# ------------------------------------------------------------------------------
with torch.no_grad():
    _, _, latent_vector = encoder(input_image)  # Get latent vector
latent_vector = latent_vector.cpu().detach().numpy().reshape(-1)

print(f" Extracted Latent Vector of Shape: {latent_vector.shape}")

# ------------------------------------------------------------------------------
# Define SHAP Model Function
# ------------------------------------------------------------------------------
def model_fn(latent_vectors):
    latent_tensors = torch.tensor(latent_vectors, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed_images = decoder(latent_tensors)  # Pass through decoder
    return reconstructed_images.cpu().detach().numpy()

# ------------------------------------------------------------------------------
# Compute SHAP Values
# ------------------------------------------------------------------------------
print(" Computing SHAP values... (This may take a few minutes)")
explainer = shap.Explainer(model_fn, latent_vector.reshape(1, -1))
shap_values = explainer(latent_vector.reshape(1, -1))  # Explain single sample

# ------------------------------------------------------------------------------
# Rank Latent Dimensions by Importance
# ------------------------------------------------------------------------------
feature_importance = np.abs(shap_values.values).sum(axis=0)
important_features = np.argsort(feature_importance)[::-1]  # Most important first

print(f" Ranked Latent Features (Top 5): {important_features[:5]}")

# ------------------------------------------------------------------------------
# Plot Cumulative Contribution of Features
# ------------------------------------------------------------------------------
cumulative_importance = np.cumsum(feature_importance[important_features]) / np.sum(feature_importance)

plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance * 100)
plt.xlabel("Number of Latent Features")
plt.ylabel("Cumulative Contribution (%)")
plt.title("Contribution of Latent Features to Reconstruction")
plt.grid()
plt.axvline(x=NEW_LATENT_DIM, color='r', linestyle='--', label=f"Target: {NEW_LATENT_DIM} Features")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# Select the Top N Features for New Latent Space
# ------------------------------------------------------------------------------
top_features = important_features[:NEW_LATENT_DIM]
print(f" Selected Top {NEW_LATENT_DIM} Features: {top_features}")

# ------------------------------------------------------------------------------
# Modify Encoder to Use Reduced Latent Space
# ------------------------------------------------------------------------------
class ReducedEncoder(torch.nn.Module):
    def __init__(self, original_encoder, selected_features):
        super(ReducedEncoder, self).__init__()
        self.original_encoder = original_encoder
        self.selected_features = selected_features  # Only keep top features

    def forward(self, x):
        mean, logvar, latent_vector = self.original_encoder(x)
        reduced_latent_vector = latent_vector[:, self.selected_features]  # Select important features
        return mean, logvar, reduced_latent_vector

# ------------------------------------------------------------------------------
# Modify Decoder to Accept New Latent Dimension
# ------------------------------------------------------------------------------
class ReducedDecoder(torch.nn.Module):
    def __init__(self, original_decoder, selected_features, full_dim):
        super(ReducedDecoder, self).__init__()
        self.original_decoder = original_decoder
        self.selected_features = selected_features
        self.full_dim = full_dim

    def forward(self, reduced_latent_vector):
        # Create a full-sized latent vector, setting unselected features to zero
        full_latent_vector = torch.zeros((reduced_latent_vector.shape[0], self.full_dim)).to(device)
        full_latent_vector[:, self.selected_features] = reduced_latent_vector
        return self.original_decoder(full_latent_vector)

# ------------------------------------------------------------------------------
# Instantiate New Models and Save Them
# ------------------------------------------------------------------------------
reduced_encoder = ReducedEncoder(encoder, top_features).to(device)
reduced_decoder = ReducedDecoder(decoder, top_features, LATENT_DIM).to(device)

print(f"✅ New Models Created with Latent Dimension: {NEW_LATENT_DIM}")

# Save the models
REDUCED_ENCODER_PATH = "model/reduced_encoder.pth"
REDUCED_DECODER_PATH = "model/reduced_decoder.pth"

torch.save(reduced_encoder.state_dict(), REDUCED_ENCODER_PATH)
torch.save(reduced_decoder.state_dict(), REDUCED_DECODER_PATH)

print(f" Saved Reduced Encoder: {REDUCED_ENCODER_PATH}")
print(f" Saved Reduced Decoder: {REDUCED_DECODER_PATH}")

print(" Reduced Latent Space Model Ready! Next Step: Train with Reduced Features.")
