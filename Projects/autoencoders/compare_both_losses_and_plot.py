import pandas as pd
import matplotlib.pyplot as plt
import wandb

# ------------------------------
# Font & Style Configuration
# ------------------------------
import matplotlib
matplotlib.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# ------------------------------
# Load Training Logs
# ------------------------------
logcosh_df = pd.read_csv("logs/200_epochs_128_ls_logcosh_training_log.csv")
mse_df = pd.read_csv("logs/200_epochs_128_ls_mse_training_log.csv")

print("Log-Cosh Loss Summary:\n", logcosh_df.describe())
print("\nMSE Loss Summary:\n", mse_df.describe())

# ------------------------------
# Initialize W&B
# ------------------------------
wandb.init(project="vae-loss-comparison", name="128_latent_loss_compare", reinit=True)

# ------------------------------
# Plot Utility
# ------------------------------
def plot_and_log(x, y1, y2, label1, label2, ylabel, title, filename, wandb_key):
    plt.figure(figsize=(10, 8))
    plt.plot(x, y1, label=label1, marker='o')
    plt.plot(x, y2, label=label2, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = f"logs/{filename}"
    plt.savefig(path)
    wandb.log({wandb_key: wandb.Image(path)})
    plt.close()

# ------------------------------
# Plot: Validation Loss
# ------------------------------
plot_and_log(
    x=logcosh_df["Epoch"],
    y1=logcosh_df["Val Loss"],
    y2=mse_df["Val Loss"],
    label1="Log-Cosh Loss",
    label2="MSE Loss",
    ylabel="Validation Loss",
    title="Validation Loss Comparison",
    filename="val_loss_comparison.png",
    wandb_key="Validation Loss Comparison"
)

# ------------------------------
# Plot: PSNR
# ------------------------------
if "Val PSNR" in logcosh_df.columns and "Val PSNR" in mse_df.columns:
    plot_and_log(
        x=logcosh_df["Epoch"],
        y1=logcosh_df["Val PSNR"],
        y2=mse_df["Val PSNR"],
        label1="Log-Cosh PSNR",
        label2="MSE PSNR",
        ylabel="Validation PSNR (dB)",
        title="Validation PSNR Comparison",
        filename="val_psnr_comparison.png",
        wandb_key="Validation PSNR Comparison"
    )

# ------------------------------
# Plot: SSIM
# ------------------------------
if "Val SSIM" in logcosh_df.columns and "Val SSIM" in mse_df.columns:
    plot_and_log(
        x=logcosh_df["Epoch"],
        y1=logcosh_df["Val SSIM"],
        y2=mse_df["Val SSIM"],
        label1="Log-Cosh SSIM",
        label2="MSE SSIM",
        ylabel="Validation SSIM",
        title="Validation SSIM Comparison",
        filename="val_ssim_comparison.png",
        wandb_key="Validation SSIM Comparison"
    )

# ------------------------------
# Optionally Log Summary Metrics
# ------------------------------
wandb.log({
    "Best Val Loss (Log-Cosh)": logcosh_df["Val Loss"].min(),
    "Best Val Loss (MSE)": mse_df["Val Loss"].min(),
    "Best PSNR (Log-Cosh)": logcosh_df.get("Val PSNR", pd.Series()).max(),
    "Best PSNR (MSE)": mse_df.get("Val PSNR", pd.Series()).max(),
    "Best SSIM (Log-Cosh)": logcosh_df.get("Val SSIM", pd.Series()).max(),
    "Best SSIM (MSE)": mse_df.get("Val SSIM", pd.Series()).max(),
})

wandb.finish()
