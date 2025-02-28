import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------------
# Define Class Types to Process (2_class and 4_class)
# ------------------------------------------------------------------------------
class_types = ["2_class", "4_class"]

# Loop through each class type
for class_type in class_types:
    # Determine the suffix used in file names for this class type
    class_suffix = "2_classes" if class_type == "2_class" else "4_classes"

    # ------------------------------------------------------------------------------
    # Setup: Define Paths & Create Output Directories
    # ------------------------------------------------------------------------------
    # Define file paths for different methods based on the class type
    METHODS_FILES = {
        "Grid-Based Masking": f"results/masking/grid_based/grid_based_masking_{class_suffix}_results.csv",
        "Object Detection": f"results/masking/object_detection/object_detection_masking_{class_suffix}_results.csv",
        "LIME on Images": f"results/masking/lime_on_images/lime_on_image_masking_{class_suffix}_results.csv",
        "LIME on Latent Features": f"results/masking/lime_on_latent/lime_on_latent_masking_{class_suffix}_results.csv"
    }

    # Define output directories for plots and results specific to the class type
    PLOTS_DIR = os.path.join("plots", "sanity_check", class_type)
    RESULTS_DIR = os.path.join("results", "sanity_check", class_type)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------------------
    # Load Data
    # ------------------------------------------------------------------------------
    df_list = []
    for method, filepath in METHODS_FILES.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["Method"] = method  # add method column
            df_list.append(df)
        else:
            print(f"⚠ Warning: {filepath} not found. Skipping...")
    
    if not df_list:
        print(f"No data loaded for {class_type}. Skipping this class type.")
        continue

    # Merge data from all methods
    df = pd.concat(df_list, ignore_index=True)

    # ------------------------------------------------------------------------------
    # Data Preprocessing
    # ------------------------------------------------------------------------------
    # Select only the relevant columns
    df = df[["Method", "Image File", "Counterfactual Found", "PSNR", "SSIM", "MSE", "UQI", "VIFP"]]

    # Filter rows where Counterfactual Found == True
    df_true = df[df["Counterfactual Found"] == True]

    # ------------------------------------------------------------------------------
    # Function: Save Boxplots (with Legends)
    # ------------------------------------------------------------------------------
    def save_boxplot(metric):
        """
        Generates and saves a boxplot for a given metric comparing methods.
        
        Args:
            metric (str): The metric to plot (e.g., "PSNR", "SSIM").
        """
        plt.figure(figsize=(10, 6))
        # Use 'Counterfactual Found' as the x-axis (will be True for all rows)
        # and use hue="Method" to display the different methods along with a legend.
        sns.boxplot(data=df_true, x="Counterfactual Found", y=metric, hue="Method")
        plt.title(f"{metric} vs. Counterfactual Found for {class_type}", fontsize=14)
        plt.xlabel("Counterfactual Found", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend(title="Method", fontsize=10)
        plot_path = os.path.join(PLOTS_DIR, f"{metric.lower()}_boxplot_{class_type}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f" {metric} boxplot saved at: {plot_path}")

    # Generate and save boxplots for each metric
    for metric in ["PSNR", "SSIM", "MSE", "UQI", "VIFP"]:
        save_boxplot(metric)

    # ------------------------------------------------------------------------------
    # Compute Summary Statistics (Only for True Cases)
    # ------------------------------------------------------------------------------
    summary = df_true.groupby("Method").agg({
        "PSNR": ["mean", "std", "min", "max"],
        "SSIM": ["mean", "std", "min", "max"],
        "MSE": ["mean", "std", "min", "max"],
        "UQI": ["mean", "std", "min", "max"],
        "VIFP": ["mean", "std", "min", "max"]
    }).reset_index()

    # Flatten MultiIndex columns
    summary.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in summary.columns]

    # ------------------------------------------------------------------------------
    # Save Summary as CSV
    # ------------------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, f"sanity_check_image_vs_ce_metrics_{class_type}.csv")
    summary.to_csv(csv_path, index=False)
    print(f"Sanity check summary saved to:\n CSV: {csv_path}\n")
