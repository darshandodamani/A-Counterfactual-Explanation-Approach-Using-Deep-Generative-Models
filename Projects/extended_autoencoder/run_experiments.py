#!/usr/bin/env python3
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import your model and dataset components.
# These files should be in your PYTHONPATH or same directory.
from encoder import VariationalEncoder
from decoder import Decoder

# We re-use the VAE training model from our previous code.
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, num_epochs):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, num_epochs)
        self.decoder = Decoder(latent_dims, num_epochs)
        self.model_file = os.path.join(
            f"model/epochs_{num_epochs}_latent_{latent_dims}/",
            "var_autoencoder.pth"
        )
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()
        print(f"VAE model saved to {self.model_file}")

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        self.encoder.load()
        self.decoder.load()
        print(f"VAE model loaded from {self.model_file}")

# Custom Dataset (update paths as needed)
class CustomImageDatasetWithLabels(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                img_path = os.path.join(img_dir, row["image_filename"])
                label = 0 if row["label"] == "STOP" else 1
                self.data.append((img_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# Loss Functions
def log_cosh_loss(x, recon_x, a=1.0, reduction="sum"):
    diff = recon_x - x
    loss = (1.0 / a) * torch.log(torch.cosh(a * diff))
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def safe_log_cosh_loss(x, recon_x, a=1.0, reduction="sum"):
    # Ensure that recon_x matches x spatially before computing loss.
    if recon_x.shape[2:] != x.shape[2:]:
        recon_x = F.interpolate(recon_x, size=x.shape[2:], mode="bilinear", align_corners=False)
    return log_cosh_loss(x, recon_x, a=a, reduction=reduction)

def vae_loss(x, recon_x, mu, logvar, a=1.0, kl_weight=1.0):
    recon_loss = safe_log_cosh_loss(x, recon_x, a=a, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kld

# Training and Validation functions
def train_epoch(model, trainloader, optimizer, epoch, kl_weight, device):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    for batch_idx, (x, _, _) in enumerate(trainloader):
        x = x.to(device)
        recon, mu, logvar, _ = model(x)
        loss = vae_loss(x, recon, mu, logvar, a=1.0, kl_weight=kl_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += safe_log_cosh_loss(x, recon, a=1.0, reduction="sum").item()
        total_kl_loss += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(trainloader.dataset)
    avg_recon = total_recon_loss / len(trainloader.dataset)
    avg_kl = total_kl_loss / len(trainloader.dataset)
    return avg_loss, avg_recon, avg_kl

def validate_epoch(model, valloader, kl_weight, device):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    with torch.no_grad():
        for x, _, _ in valloader:
            x = x.to(device)
            recon, mu, logvar, _ = model(x)
            loss = vae_loss(x, recon, mu, logvar, a=1.0, kl_weight=kl_weight)
            total_loss += loss.item()
            total_recon_loss += safe_log_cosh_loss(x, recon, a=1.0, reduction="sum").item()
            total_kl_loss += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()
    avg_loss = total_loss / len(valloader.dataset)
    avg_recon = total_recon_loss / len(valloader.dataset)
    avg_kl = total_kl_loss / len(valloader.dataset)
    return avg_loss, avg_recon, avg_kl

# Experiment function: Runs training for one configuration and returns best validation loss
def run_training_experiment(config, device):
    # Unpack hyperparameters
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    kl_weight_initial = config["kl_weight_initial"]
    latent_space = config["latent_space"]
    
    # Data transforms and datasets
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])
    data_dir = "dataset/town7_dataset/"
    train_csv = os.path.join(data_dir, "train", "labeled_train_data_log.csv")
    test_csv = os.path.join(data_dir, "test", "labeled_test_data_log.csv")
    
    train_data = CustomImageDatasetWithLabels(os.path.join(data_dir, "train"), train_csv, transform=train_transforms)
    # We'll use part of train_data for validation (80/20 split)
    train_len = int(0.8 * len(train_data))
    val_len = len(train_data) - train_len
    train_subset, val_subset = random_split(train_data, [train_len, val_len])
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = VariationalAutoencoder(latent_dims=latent_space, num_epochs=num_epochs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=False)
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 20  # you can adjust patience for early stopping
    
    for epoch in range(1, num_epochs + 1):
        kl_weight = min(kl_weight_initial + epoch * 0.0001, 1.0)
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, epoch, kl_weight, device)
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, kl_weight, device)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | Val Loss={val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
    
    return best_val_loss

###############################################
# Main Experiment Loop
###############################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define hyperparameter grid
    learning_rates = [1e-4, 5e-5]
    batch_sizes = [128, 256]
    kl_weight_initials = [0.00005, 0.0001]
    num_epochs_list = [100, 200]  # For initial experiments, use fewer epochs
    latent_space = 128  # We'll keep this fixed as per your design
    
    # Create a directory to save experiment results
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "results.csv")
    
    # Write header to CSV
    with open(results_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Learning Rate", "Batch Size", "KL Weight Init", "Num Epochs", "Best Val Loss"])
    
    # Loop over hyperparameter combinations
    for lr in learning_rates:
        for bs in batch_sizes:
            for kl_init in kl_weight_initials:
                for epochs in num_epochs_list:
                    config = {
                        "learning_rate": lr,
                        "batch_size": bs,
                        "kl_weight_initial": kl_init,
                        "num_epochs": epochs,
                        "latent_space": latent_space,
                    }
                    print("\nRunning experiment with config:", config)
                    best_val_loss = run_training_experiment(config, device)
                    print("Best validation loss:", best_val_loss)
                    # Save result in CSV
                    with open(results_file, "a", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([lr, bs, kl_init, epochs, best_val_loss])
    
    print("All experiments completed. Results saved in", results_file)

if __name__ == "__main__":
    main()
