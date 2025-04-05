import os
import csv
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# Custom Dataset Class
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
        return image, label

# Function to calculate mean and std
def calculate_mean_std(dataset, csv_file, plot_file=None):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("Calculating mean and standard deviation...")
    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)  # Flatten HxW
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    # Save to CSV
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Channel", "Mean", "Std"])
        for i, (m, s) in enumerate(zip(mean, std)):
            writer.writerow([f"Channel {i+1}", m.item(), s.item()])
    
    print(f"Mean and standard deviation saved to {csv_file}")
    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")
    
    # Optional: Plot mean and std
    if plot_file:
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        channels = ["R", "G", "B"]
        plt.bar(channels, mean.tolist(), yerr=std.tolist(), capsize=5, color=["red", "green", "blue"])
        plt.xlabel("Channels")
        plt.ylabel("Values")
        plt.title("Mean and Standard Deviation per Channel")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot saved to {plot_file}")
    
    return mean, std

if __name__ == "__main__":
    # Paths to dataset and CSV files
    data_dir = "dataset/town7_dataset/"
    train_csv = os.path.join(data_dir, "train", "labeled_train_4_class_data_log.csv")
    test_csv = os.path.join(data_dir, "test", "labeled_test_4_class_data_log.csv")
    
    # Transforms for calculation (no normalization)
    transforms_for_calc = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])
    
    # Datasets
    train_data = CustomImageDatasetWithLabels(
        os.path.join(data_dir, "train"), train_csv, transform=transforms_for_calc
    )
    test_data = CustomImageDatasetWithLabels(
        os.path.join(data_dir, "test"), test_csv, transform=transforms_for_calc
    )
    
    # Calculate and save mean and std
    train_mean, train_std = calculate_mean_std(
        train_data, 
        csv_file="plots/mean_std/train_mean_std.csv", 
        plot_file="plots/mean_std/train_mean_std_plot.png"
    )
    test_mean, test_std = calculate_mean_std(
        test_data, 
        csv_file="plots/mean_std/test_mean_std.csv", 
        plot_file="plots/mean_std/test_mean_std_plot.png"
    )
    
    # Print values for reference
    print("Train Dataset Mean:", train_mean.tolist())
    print("Train Dataset Std:", train_std.tolist())
    print("Test Dataset Mean:", test_mean.tolist())
    print("Test Dataset Std:", test_std.tolist())
