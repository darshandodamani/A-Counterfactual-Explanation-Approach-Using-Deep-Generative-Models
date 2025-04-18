import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn import linear_model
from vae import (
    LATENT_SPACE,
    NUM_EPOCHS,
    VariationalAutoencoder,
    CustomImageDatasetWithLabels,
)
from torchvision.transforms import transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Hyperparameters
input_size = 128
num_epochs = 20
learning_rate = 0.001
batch_size = 128

# Load the trained VAE model
vae_model = VariationalAutoencoder(latent_dims=input_size, num_epochs=NUM_EPOCHS).to(device)  # Instantiate the VAE model with latent dimensions of 128
vae_model.load_state_dict(
    torch.load(
        "model/epochs_500_latent_128_town_7/var_autoencoder.pth",
        map_location=device,
        weights_only=True,
    )
)
print("VAE model loaded successfully!")
vae_model.eval()

# Load the test dataset
data_dir = "dataset/town7_dataset/train/"
csv_file = "dataset/town7_dataset/train/labeled_train_data_log.csv"
data_transforms = transforms.Compose([transforms.ToTensor()])

# Create the test dataset and data loader
test_dataset = CustomImageDatasetWithLabels(
    img_dir=data_dir, csv_file=csv_file, transform=data_transforms
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Extract latent vectors and corresponding labels from the test dataset
latent_vectors = []
labels = []

with torch.no_grad():  # Disable gradient computation for inference
    for images, label_batch, _ in test_loader:
        images = images.to(device)

        # Encode images into latent space using the VAE encoder
        _, _, latent_space = vae_model.encoder(images)
        latent_vectors.append(latent_space.cpu().numpy())
        labels.append(label_batch.cpu().numpy())

latent_vectors = np.vstack(latent_vectors)
labels = np.hstack(labels)

# Print total number of samples in the dataset
print(f"Total number of samples in the dataset: {len(latent_vectors)}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(latent_vectors, labels, test_size=0.2, random_state=42)  # Split data into 80% training and 20% test

# Print total number of samples in training and test set
print(f"Number of samples in training set: {len(X_train)}")
print(f"Number of samples in test set: {len(X_test)}")

# Train Logistic Regression Model
logistic_model = linear_model.LogisticRegression(max_iter=500)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred_logistic)
precision = precision_score(y_test, y_pred_logistic, average='weighted')
recall = recall_score(y_test, y_pred_logistic, average='weighted')
f1 = f1_score(y_test, y_pred_logistic, average='weighted')

# Print evaluation metrics
print(f"Logistic Regression - Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_logistic)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["STOP", "GO"], yticklabels=["STOP", "GO"])  # Plot the confusion matrix as a heatmap
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("plots/logistic_regression_confusion_matrix.png")
plt.show()

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_logistic, pos_label=1)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) - Logistic Regression")
plt.legend(loc="lower right")
plt.savefig("plots/logistic_regression_roc_curve.png")
plt.show()

# Confusion Matrix Details
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"True Positives (TP): {conf_matrix[1][1]}")
print(f"False Positives (FP): {conf_matrix[0][1]}")
print(f"True Negatives (TN): {conf_matrix[0][0]}")
print(f"False Negatives (FN): {conf_matrix[1][0]}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Save the trained logistic regression model
torch.save(logistic_model, "model/epochs_500_latent_128/logistic_regression_model.pth")

# Save all the evaluation metrics and plots in the plots/classifier_plots directory
os.makedirs("plots/classifier_plots/", exist_ok=True)
plt.savefig("plots/classifier_plots/logistic_regression_confusion_matrix.png")
plt.savefig("plots/classifier_plots/logistic_regression_roc_curve.png")
print("Logistic Regression model saved successfully!")

# Note: Logistic regression does not have epoch-based training, so train_losses and train_accuracies are not applicable.
# Save the placeholder training plots and loss and accuracy in the classifier_training_loss_accuracy.png
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot([], label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([], label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("classifier_training_loss_accuracy.png")
