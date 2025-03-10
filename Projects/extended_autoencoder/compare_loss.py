import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV logs
logcosh_df = pd.read_csv("logs/200_epochs_128_ls_logcosh_training_log.csv")
mse_df = pd.read_csv("logs/200_epochs_128_ls_mse_training_log.csv")

# Print summary statistics
print("Log-Cosh Loss Log:")
print(logcosh_df.describe())
print("\nMSE Loss Log:")
print(mse_df.describe())

# Plot validation loss comparison
plt.figure(figsize=(10, 6))
plt.plot(logcosh_df["Epoch"], logcosh_df["Val Loss"], label="Log-Cosh Loss", marker='o')
plt.plot(mse_df["Epoch"], mse_df["Val Loss"], label="MSE Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("logs/val_loss_comparison.png")
plt.show()

# Plot PSNR comparison (if you logged it)
if "Val PSNR" in logcosh_df.columns and "Val PSNR" in mse_df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(logcosh_df["Epoch"], logcosh_df["Val PSNR"], label="Log-Cosh Loss", marker='o')
    plt.plot(mse_df["Epoch"], mse_df["Val PSNR"], label="MSE Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Validation PSNR (dB)")
    plt.title("Validation PSNR Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/val_psnr_comparison.png")
    plt.show()
