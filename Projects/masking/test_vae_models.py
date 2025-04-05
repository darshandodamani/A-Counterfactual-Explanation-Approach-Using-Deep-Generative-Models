import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import logging

# Add the parent directory to sys.path to import from masking_utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

# Import utilities from masking_utils
from masking_utils import load_models, save_images_without_mask

# ----------------------------------------------------------------------------
# Setup Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# Paths and Constants
# ----------------------------------------------------------------------------
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_vae_models(image_path: str, classifier_type: str = "4_class"):
    """
    Loads VAE models, performs initial prediction and classification, and saves original and reconstructed images.

    Args:
        image_path (str): Path to the image file to be tested.
        classifier_type (str): "2_class" or "4_class" to specify which classifier to use.
    """
    logging.info(f"Testing VAE models with {classifier_type} classification.")

    # Load correct models based on classifier type
    encoder, decoder, classifier = load_models(classifier_type)
    logging.info(f"Models loaded for {classifier_type}.")
    
    # Define label mappings
    CLASS_LABELS_4_CLASS = {0: "STOP", 1: "GO", 2: "RIGHT", 3: "LEFT"}
    CLASS_LABELS_2_CLASS = {0: "STOP", 1: "GO"}
    label_mapping = CLASS_LABELS_2_CLASS if classifier_type == "2_class" else CLASS_LABELS_4_CLASS
    
    IMAGE_DIRS = {
        "original": f"plots/test_vae_{classifier_type}_original",
        "reconstructed": f"plots/test_vae_{classifier_type}_reconstructed"
    }
    
    for dir_path in IMAGE_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return

    logging.info(f"Processing image: {os.path.basename(image_path)}")
    input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

    # Step 1: Encode the image
    latent_vector = encoder(input_image)[2]

    # Step 2: Decode the latent vector to reconstruct the image
    reconstructed_image = decoder(latent_vector)
    reconstructed_image = torch.nn.functional.interpolate(reconstructed_image, size=(input_image.shape[2], input_image.shape[3]), mode="bilinear", align_corners=False)

    # Step 3: Classify the latent vector
    prediction = classifier(latent_vector)
    predicted_label_idx = torch.argmax(prediction, dim=1).item()
    predicted_label = label_mapping[predicted_label_idx]
    confidence = [round(float(x), 5) for x in torch.softmax(prediction, dim=1).cpu().detach().numpy().flatten()]

    logging.info(f"Prediction: {predicted_label}")
    logging.info(f"Confidence: {confidence}")

    # Step 4: Save original and reconstructed images
    save_images_without_mask(
        os.path.basename(image_path), input_image, reconstructed_image, IMAGE_DIRS
    )
    logging.info("Original and reconstructed images saved.")

if __name__ == "__main__":
    # Example usage
    image_path = "dataset/town7_dataset/test/town7_000269.png"  # Replace with the actual path to your test image
    classifier_type = "4_class"  # or "2_class"
    test_vae_models(image_path, classifier_type)