#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F

# Import the updated encoder and decoder.
from encoder import VariationalEncoder
from decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################
# Define the VAE Model
###############################################
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
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        self.encoder.load()
        self.decoder.load()
        print(f"VAE model loaded from {self.model_file}")

###############################################
# Custom Dataset Class
###############################################
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

###############################################
# Loss Functions and Helpers
###############################################
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
    """
    Ensure recon_x has the same spatial dimensions as x before computing log-cosh loss.
    """
    if recon_x.shape[2:] != x.shape[2:]:
        recon_x = F.interpolate(recon_x, size=x.shape[2:], mode="bilinear", align_corners=False)
    return log_cosh_loss(x, recon_x, a=a, reduction=reduction)

def vae_loss(x, recon_x, mu, logvar, a=1.0, kl_weight=1.0):
    recon_loss = safe_log_cosh_loss(x, recon_x, a=a, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kld

###############################################
# Training and Validation Functions
###############################################
def train(model, trainloader, optimizer, epoch, kl_weight):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    for batch_idx, (x, _, _) in enumerate(trainloader):
        x = x.to(device)
        recon, mu, logvar, z = model(x)
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

def validate(model, valloader, kl_weight):
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    with torch.no_grad():
        for x, _, _ in valloader:
            x = x.to(device)
            recon, mu, logvar, z = model(x)
            loss = vae_loss(x, recon, mu, logvar, a=1.0, kl_weight=kl_weight)
            total_loss += loss.item()
            total_recon_loss += safe_log_cosh_loss(x, recon, a=1.0, reduction="sum").item()
            total_kl_loss += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()
    avg_loss = total_loss / len(valloader.dataset)
    avg_recon = total_recon_loss / len(valloader.dataset)
    avg_kl = total_kl_loss / len(valloader.dataset)
    return avg_loss, avg_recon, avg_kl

###############################################
# Plot Loss Curves
###############################################
def plot_losses(train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses, num_epochs):
    epochs = list(range(1, num_epochs+1))
    plt.figure(figsize=(12, 4))
    
    # Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Total Loss")
    plt.legend()
    
    # Reconstruction Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_recon_losses, label="Train Recon Loss")
    plt.plot(epochs, val_recon_losses, label="Val Recon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Recon Loss")
    plt.title("Reconstruction Loss (log-cosh)")
    plt.legend()
    
    # KL Loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_kl_losses, label="Train KL Loss")
    plt.plot(epochs, val_kl_losses, label="Val KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title("KL Divergence")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("vae_loss_plots.png")
    plt.show()

###############################################
# MAIN TRAINING LOOP
###############################################
def main():
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    LATENT_SPACE = 128
    KL_WEIGHT_INITIAL = 0.00005
    
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
    test_data = CustomImageDatasetWithLabels(os.path.join(data_dir, "test"), test_csv, transform=test_transforms)
    
    train_len = int(0.8 * len(train_data))
    val_len = len(train_data) - train_len
    train_subset, val_subset = random_split(train_data, [train_len, val_len])
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE, num_epochs=NUM_EPOCHS).to(device)
    print("Initialized VAE with latent space size:", LATENT_SPACE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    train_recon_losses = []
    val_recon_losses = []
    train_kl_losses = []
    val_kl_losses = []
    
    kl_weight = KL_WEIGHT_INITIAL
    patience = 50
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        kl_weight = min(KL_WEIGHT_INITIAL + epoch * 0.0001, 1.0)
        
        train_loss, train_recon, train_kl = train(model, train_loader, optimizer, epoch, kl_weight)
        val_loss, val_recon, val_kl = validate(model, val_loader, kl_weight)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_recon_losses.append(train_recon)
        val_recon_losses.append(val_recon)
        train_kl_losses.append(train_kl)
        val_kl_losses.append(val_kl)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"           Recon: Train={train_recon:.4f}, Val={val_recon:.4f} | KL: Train={train_kl:.4f}, Val={val_kl:.4f}")
        
        if epoch % 50 == 0:
            checkpoint_path = f"model/{NUM_EPOCHS}_epochs_LATENT_{LATENT_SPACE}/checkpoint_epoch_{epoch}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    plot_losses(train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses, epoch)
    
    model.save()
    print("Training complete.")

if __name__ == "__main__":
    main()
