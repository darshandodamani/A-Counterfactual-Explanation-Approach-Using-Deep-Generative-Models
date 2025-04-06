import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visuals
sns.set(style="whitegrid")

# Metrics and class types
metrics_to_plot = ["SSIM", "PSNR", "MSE"]
class_types = ["2_class", "4_class"]

# Define method paths
def get_methods_files(class_suffix):
    return {
        "Grid-Based Masking": f"results/masking/grid_based/grid_based_masking_{class_suffix}_results.csv",
        "Object Detection": f"results/masking/object_detection/object_detection_masking_{class_suffix}_results.csv",
        "LIME on Images": f"results/masking/lime_on_images/lime_on_image_masking_{class_suffix}_results.csv",
        "LIME on Latent Features": f"results/masking/lime_on_latent/lime_on_latent_masking_{class_suffix}_results.csv",
        "LIME on Latent NUN": f"results/masking/lime_on_latent/{class_suffix}_NUN_results.csv"
    }

# Process each class type
for class_type in class_types:
    class_suffix = "2_classes" if class_type == "2_class" else "4_classes"
    methods_files = get_methods_files(class_suffix)

    # Output directory
    plots_dir = os.path.join("plots", "sanity_check", class_type)
    os.makedirs(plots_dir, exist_ok=True)

    # Load data
    df_list = []
    for method, filepath in methods_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["Method"] = method
            df_list.append(df)

    if not df_list:
        print(f"No valid files found for {class_type}. Skipping...")
        continue

    df_combined = pd.concat(df_list, ignore_index=True)
    df_true = df_combined[df_combined["Counterfactual Found"] == True]

    if df_true.empty:
        print(f"No counterfactuals found for {class_type}. Skipping...")
        continue

    # Generate boxplots
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(
            data=df_true,
            x="Counterfactual Found",
            y=metric,
            hue="Method",
            showfliers=False  # Remove small outlier circles
        )
        ax.set_title(f"{metric} Comparison Across Methods ({class_type.replace('_', ' ').title()})", fontsize=16)
        ax.set_xlabel("Counterfactual Found", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title="Method", fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(plots_dir, f"{metric.lower()}_boxplot_{class_type}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved: {plot_path}")
