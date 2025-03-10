#!/usr/bin/env python3
import os
import sys
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch.nn.functional as F

# Import your updated encoder and decoder modules.
from encoder import VariationalEncoder
from decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#----------------------------------------------
# Configuration
#----------------------------------------------
NUM_EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
LATENT_SPACE = 128
KL_WEIGHT_INITIAL = 5e-05

# Choose loss type: "logcosh" or "mse"
loss_type = "logcosh"

# Dynamic folder name based on loss type.
config_folder = f"200_epochs_{LATENT_SPACE}_ls_{loss_type}"

#----------------------------------------------
# Define the VAE Model
#----------------------------------------------
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, num_epochs):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, num_epochs)
        self.decoder = Decoder(latent_dims, num_epochs)
        # Include loss type in folder name.
        self.model_file = os.path.join(
            f"model/epochs_{num_epochs}_latent_{latent_dims}_ls_{loss_type}/",
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

#----------------------------------------------
# Custom Dataset Class
#----------------------------------------------
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

#----------------------------------------------
# Loss Functions and Loss Selector
#----------------------------------------------
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
    if recon_x.shape[2:] != x.shape[2:]:
        recon_x = F.interpolate(recon_x, size=x.shape[2:], mode="bilinear", align_corners=False)
    return log_cosh_loss(x, recon_x, a=a, reduction=reduction)

def mse_loss_fn(x, recon_x, reduction="sum"):
    if recon_x.shape[2:] != x.shape[2:]:
        recon_x = F.interpolate(recon_x, size=x.shape[2:], mode="bilinear", align_corners=False)
    return torch.nn.functional.mse_loss(recon_x, x, reduction=reduction)

def vae_loss_selected(x, recon_x, mu, logvar, kl_weight=1.0, a=1.0):
    if loss_type == "logcosh":
        recon_loss = safe_log_cosh_loss(x, recon_x, a=a, reduction="sum")
    elif loss_type == "mse":
        recon_loss = mse_loss_fn(x, recon_x, reduction="sum")
    else:
        raise ValueError("Unknown loss type")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kld

#----------------------------------------------
# Additional Metrics: Pixel Accuracy and PSNR
#----------------------------------------------
def pixel_accuracy(x, recon_x, threshold=0.5):
    if recon_x.shape[2:] != x.shape[2:]:
        recon_x = F.interpolate(recon_x, size=x.shape[2:], mode="bilinear", align_corners=False)
    x_bin = (x > threshold).float()
    recon_bin = (recon_x > threshold).float()
    correct = (x_bin == recon_bin).float().sum()
    total = torch.numel(x_bin)
    return correct / total

def calculate_psnr(x, recon_x):
    if recon_x.shape[2:] != x.shape[2:]:
        recon_x = F.interpolate(recon_x, size=x.shape[2:], mode="bilinear", align_corners=False)
    mse = torch.nn.functional.mse_loss(recon_x, x, reduction="mean").item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)

#----------------------------------------------
# Function to Save Sample Reconstructions
#----------------------------------------------
def save_reconstruction_sample(model, dataloader, epoch, device, folder=f"plots/{config_folder}/reconstructions"):
    model.eval()
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        for x, _, _ in dataloader:
            x = x.to(device)
            recon, _, _, _ = model(x)
            if recon.shape[2:] != x.shape[2:]:
                recon = F.interpolate(recon, size=x.shape[2:], mode="bilinear", align_corners=False)
            orig_img = x[0].cpu().permute(1,2,0).numpy()
            recon_img = recon[0].cpu().permute(1,2,0).numpy()
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.imshow(orig_img)
            plt.title("Original")
            plt.axis("off")
            plt.subplot(1,2,2)
            plt.imshow(recon_img)
            plt.title("Reconstruction")
            plt.axis("off")
            plt.tight_layout()
            save_path = os.path.join(folder, f"epoch_{epoch}.png")
            plt.savefig(save_path)
            plt.close()
            break
    print(f"Reconstruction sample saved at epoch {epoch} in {folder}")

#----------------------------------------------
# Training and Validation Functions
#----------------------------------------------
def train_epoch(model, trainloader, optimizer, epoch, kl_weight, device):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    for batch_idx, (x, _, _) in enumerate(trainloader):
        x = x.to(device)
        recon, mu, logvar, _ = model(x)
        loss = vae_loss_selected(x, recon, mu, logvar, kl_weight=kl_weight, a=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if loss_type == "logcosh":
            total_recon_loss += safe_log_cosh_loss(x, recon, a=1.0, reduction="sum").item()
        else:
            total_recon_loss += mse_loss_fn(x, recon, reduction="sum").item()
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
    acc_sum = 0.0
    psnr_sum = 0.0
    total_samples = 0
    with torch.no_grad():
        for x, _, _ in valloader:
            x = x.to(device)
            recon, mu, logvar, _ = model(x)
            if recon.shape[2:] != x.shape[2:]:
                recon = F.interpolate(recon, size=x.shape[2:], mode="bilinear", align_corners=False)
            loss = vae_loss_selected(x, recon, mu, logvar, kl_weight=kl_weight, a=1.0)
            total_loss += loss.item()
            if loss_type == "logcosh":
                total_recon_loss += safe_log_cosh_loss(x, recon, a=1.0, reduction="sum").item()
            else:
                total_recon_loss += mse_loss_fn(x, recon, reduction="sum").item()
            total_kl_loss += (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()
            acc = pixel_accuracy(x, recon, threshold=0.5)
            acc_sum += acc.item() * x.size(0)
            batch_psnr = 0.0
            for i in range(x.size(0)):
                mse_val = torch.nn.functional.mse_loss(recon[i:i+1], x[i:i+1], reduction="mean").item()
                batch_psnr += 10 * math.log10(1.0 / (mse_val + 1e-10))
            psnr_sum += batch_psnr
            total_samples += x.size(0)
    avg_loss = total_loss / len(valloader.dataset)
    avg_recon = total_recon_loss / len(valloader.dataset)
    avg_kl = total_kl_loss / len(valloader.dataset)
    avg_acc = acc_sum / total_samples
    avg_psnr = psnr_sum / total_samples
    return avg_loss, avg_recon, avg_kl, avg_acc, avg_psnr

#----------------------------------------------
# Plot Metrics and Save as Separate Files
#----------------------------------------------
def plot_metrics(train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses, val_accuracies, val_psnrs, num_epochs):
    plots_folder = f"plots/{config_folder}"
    os.makedirs(plots_folder, exist_ok=True)
    epochs = list(range(1, num_epochs+1))
    
    # Total Loss Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "total_loss.png"))
    plt.close()
    
    # Reconstruction Loss Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_recon_losses, label="Train Recon Loss")
    plt.plot(epochs, val_recon_losses, label="Val Recon Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss (" + loss_type.upper() + ")")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "recon_loss.png"))
    plt.close()
    
    # KL Loss Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_kl_losses, label="Train KL Loss")
    plt.plot(epochs, val_kl_losses, label="Val KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "kl_loss.png"))
    plt.close()
    
    # Validation Pixel Accuracy Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs, val_accuracies, label="Val Pixel Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Accuracy")
    plt.title("Validation Pixel Accuracy")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "val_accuracy.png"))
    plt.close()
    
    # Validation PSNR Plot
    plt.figure(figsize=(8,6))
    plt.plot(epochs, val_psnrs, label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "val_psnr.png"))
    plt.close()
    
    print(f"Metric plots saved in the '{plots_folder}' folder.")

#----------------------------------------------
# MAIN TRAINING LOOP with CSV Logging
#----------------------------------------------
def main():
    import math
    import matplotlib.ticker as mtick
    
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
    train_csv = os.path.join(data_dir, "train", "labeled_train_4_class_data_log.csv")
    test_csv = os.path.join(data_dir, "test", "labeled_test_4_class_data_log.csv")
    
    train_data = CustomImageDatasetWithLabels(os.path.join(data_dir, "train"), train_csv, transform=train_transforms)
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
    val_accuracies = []
    val_psnrs = []
    
    kl_weight = KL_WEIGHT_INITIAL
    patience = 50
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Create folders for plots and CSV logs.
    os.makedirs(f"plots/{config_folder}/reconstructions", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    csv_file = os.path.join("logs", f"{config_folder}_training_log.csv")
    
    # Write CSV header
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Train Recon Loss", "Train KL Loss", 
                         "Val Loss", "Val Recon Loss", "Val KL Loss", "Val Pixel Accuracy", "Val PSNR", "KL Weight"])
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        kl_weight = min(KL_WEIGHT_INITIAL + epoch * 0.0001, 1.0)
        
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, epoch, kl_weight, device)
        val_loss, val_recon, val_kl, val_acc, val_psnr = validate_epoch(model, val_loader, kl_weight, device)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_recon_losses.append(train_recon)
        val_recon_losses.append(val_recon)
        train_kl_losses.append(train_kl)
        val_kl_losses.append(val_kl)
        val_accuracies.append(val_acc)
        val_psnrs.append(val_psnr)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"           Recon: Train={train_recon:.4f}, Val={val_recon:.4f} | KL: Train={train_kl:.4f}, Val={val_kl:.4f}")
        print(f"           Val Pixel Accuracy: {val_acc:.4f}, Val PSNR: {val_psnr:.2f} dB")
        print(f"           KL Weight: {kl_weight:.6f}")
        
        # Log epoch details to CSV.
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_recon, train_kl, val_loss, val_recon, val_kl, val_acc, val_psnr, kl_weight])
        
        # Save sample reconstruction every 10 epochs.
        if epoch % 10 == 0:
            save_reconstruction_sample(model, val_loader, epoch, device, folder=f"plots/{config_folder}/reconstructions")
        
        if epoch % 50 == 0:
            checkpoint_path = f"model/{NUM_EPOCHS}_epochs_LATENT_{LATENT_SPACE}_{loss_type}/checkpoint_epoch_{epoch}.pth"
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

    plot_metrics(train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses, val_accuracies, val_psnrs, epoch)
    
    model.save()
    print("Training complete.")

if __name__ == "__main__":
    main()
