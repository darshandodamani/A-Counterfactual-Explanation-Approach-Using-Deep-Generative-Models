# location: Projects/autoencoder/evaluate_classifier.py
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from vae import (
    LATENT_SPACE,
    NUM_EPOCHS,
    VariationalAutoencoder,
    CustomImageDatasetWithLabels,
)
from classifier import ClassifierModel  # Import the classifier model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the trained VAE model
vae_model = VariationalAutoencoder(latent_dims=128, num_epochs=50).to(
    device
)  # Example value for num_epochs
vae_model.load_state_dict(
    torch.load(
        "model/epochs_500_latent_128_town_7/var_autoencoder.pth",
        map_location=device,
        weights_only=True,
    )
)
print("VAE model loaded successfully!")
vae_model.eval()

# Instantiate the classifier model
classifier = ClassifierModel(input_size=128, hidden_size=128, output_size=2).to(device)

# Load the classifier state dictionary
classifier.load_state_dict(
    torch.load(
        "model/epochs_500_latent_128_town_7/classifier_final.pth",
        map_location=device,
        weights_only=True,
    )
)
print("Classifier model loaded successfully!")
classifier.eval()  # Set the classifier to evaluation mode

# Load the test dataset
data_dir = "dataset/town7_dataset/test/"
print(f"Loading test dataset from: {data_dir}")
csv_file = "dataset/town7_dataset/test/labeled_test_data_log.csv"
print(f"Loading labels from: {csv_file}")
data_transforms = transforms.Compose([transforms.ToTensor()])

test_dataset = CustomImageDatasetWithLabels(
    img_dir=data_dir, csv_file=csv_file, transform=data_transforms
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Collect predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Encode images into latent space
        _, _, latent_vectors = vae_model.encoder(images)

        # Get classifier predictions on latent space
        with torch.no_grad():
            outputs = classifier(latent_vectors)
            preds = torch.argmax(
                outputs, dim=1
            )  # Get the predicted class (0 for STOP, 1 for GO)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["STOP", "GO"],
    yticklabels=["STOP", "GO"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(
    f"plots/classifier_plots/confusion_matrix_for_{NUM_EPOCHS}_epochs_{LATENT_SPACE}_LF.png"
)  # Save the plot dynamically
print(f"Confusion matrix for {NUM_EPOCHS} epochs {LATENT_SPACE} LF saved successfully!")
plt.close()

# Calculate TP, FP, TN, FN from the Confusion Matrix
tn, fp, fn, tp = conf_matrix.ravel()  # Extract TN, FP, FN, TP from the confusion matrix

# Calculate other performance metrics (optional)
accuracy = accuracy_score(all_labels, all_preds)
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
f1_score = (
    2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
)

# Compute ROC curve and AUC for class 1 (GO)
fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
roc_auc = auc(fpr, tpr)

# Save ROC Curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.savefig(
    f"plots/classifier_plots/roc_curve_for_{NUM_EPOCHS}_epochs_{LATENT_SPACE}_LF.png"
)
plt.close()

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")