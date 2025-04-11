import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO)

# ============================
# Step 0: Load and Clean Data
# ============================
file_path = "Projects/counterfactual_comparison/responses.csv"

try:
    # Use on_bad_lines='skip' to skip problematic rows
    df = pd.read_csv(file_path, on_bad_lines='skip')
    logging.info(f"CSV file loaded successfully with {len(df)} rows.")
except pd.errors.ParserError as e:
    logging.error(f"Error reading CSV file: {e}")
    raise

# Strip extra whitespace from column names
df.columns = df.columns.str.strip()

# Inspect dataset
print("Dataset Overview:\n", df.head())
print("Detected columns:", df.columns.tolist())

# ============================
# Step 1: Rename Columns (Optional)
# ============================
# If desired, use this mapping to display alternative method names.
# Update mapping to include method4.
method_mapping = {
    "Counterfactual_1_Interpretability": "Grid-Based Masking_Interpretability",
    "Counterfactual_1_Plausibility": "Grid-Based Masking_Plausibility",
    "Counterfactual_1_VisualCoherence": "Grid-Based Masking_VisualCoherence",
    
    "Counterfactual_2_Interpretability": "LIME on Images_Interpretability",
    "Counterfactual_2_Plausibility": "LIME on Images_Plausibility",
    "Counterfactual_2_VisualCoherence": "LIME on Images_VisualCoherence",
    
    "Counterfactual_3_Interpretability": "LIME on Latent Features_Interpretability",
    "Counterfactual_3_Plausibility": "LIME on Latent Features_Plausibility",
    "Counterfactual_3_VisualCoherence": "LIME on Latent Features_VisualCoherence",
    
    "Counterfactual_4_Interpretability": "LIME on Latent Feature Masking using NUN_Interpretability",
    "Counterfactual_4_Plausibility": "LIME on Latent Feature Masking using NUN_Plausibility",
    "Counterfactual_4_VisualCoherence": "LIME on Latent Feature Masking using NUN_VisualCoherence",
}

# Uncomment the following line if you want to apply renaming:
# df.rename(columns=method_mapping, inplace=True)

# ============================
# Step 2: Convert Wide Format to Long Format
# ============================
df_long = df.melt(id_vars=["Image"], var_name="Method_Criterion", value_name="Rating")

# ============================
# Step 3: Split 'Method_Criterion' into 'Method' and 'Criterion'
# Using rsplit in case method names contain underscores
# ============================
df_long["Method"], df_long["Criterion"] = zip(*df_long["Method_Criterion"].apply(lambda x: x.rsplit("_", 1)))
df_long.drop(columns=["Method_Criterion"], inplace=True)

# ============================
# Step 4: Convert Ratings to Numeric & Clean Data
# ============================
df_long["Rating"] = pd.to_numeric(df_long["Rating"], errors="coerce")
df_long.dropna(inplace=True)

# ============================
# Step 5: Aggregate and Pivot Data for Visualization
# ============================
df_summary = df_long.groupby(["Method", "Criterion"])["Rating"].mean().reset_index()
df_pivot = df_summary.pivot(index="Criterion", columns="Method", values="Rating")

# Create output directory for plots
output_dir = "plots/counterfactual_comparison/"
os.makedirs(output_dir, exist_ok=True)

# ============================
# Step 6: Combined Bar Plot Using Pivoted Data
# ============================
plt.figure(figsize=(10, 6))
df_pivot.plot(kind="bar", colormap="viridis", edgecolor="black")
plt.title("User Evaluation of Counterfactual Explanation Methods")
plt.xlabel("Evaluation Criteria")
plt.ylabel("Average Rating (1-5)")
plt.xticks(rotation=45)
plt.legend(title="Method")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
bar_plot_path = os.path.join(output_dir, "bar_plot_user_evaluations.png")
plt.savefig(bar_plot_path)
print(f"Bar plot saved: {bar_plot_path}")

# ============================
# Step 7: Heatmap Using Seaborn
# ============================
plt.figure(figsize=(12, 8))
sns.heatmap(
    df_pivot,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f",
    annot_kws={"fontsize": 14}
)
plt.title("User Evaluation Heatmap", fontsize=18)
plt.xlabel("Method", fontsize=16)
plt.ylabel("Evaluation Criterion", fontsize=16)
plt.xticks(fontsize=14, rotation=30)
plt.yticks(fontsize=14, rotation=0)
plt.tight_layout()
heatmap_path = os.path.join(output_dir, "heatmap_user_evaluations.png")
plt.savefig(heatmap_path)
print(f"Heatmap saved: {heatmap_path}")

# ============================
# Step 8: Compute Average Ratings from Wide-Format Data and Plot Individual Bar Charts
# ============================
metrics = ["Interpretability", "Plausibility", "VisualCoherence"]
# Update the methods list to include method 4
methods = [1, 2, 3, 4]
avg_ratings = {}

for metric in metrics:
    avg_ratings[metric] = []
    for method in methods:
        col_name = f"Counterfactual_{method}_{metric}"
        if col_name in df.columns:
            avg = df[col_name].mean()
            avg_ratings[metric].append(avg)
            print(f"Average {metric} for Counterfactual {method}: {avg:.2f}")
        else:
            print(f"Column {col_name} not found!")
            avg_ratings[metric].append(None)

# Plot a separate bar chart for each metric
for metric in metrics:
    plt.figure(figsize=(8, 5))
    # Filter out None values and corresponding methods
    valid_ratings = [(f"Method {m}", avg_ratings[metric][i]) for i, m in enumerate(methods) if avg_ratings[metric][i] is not None]
    methods_filtered, ratings_filtered = zip(*valid_ratings) if valid_ratings else ([], [])
    
    if ratings_filtered:  # Only plot if there are valid ratings
        plt.bar(methods_filtered, ratings_filtered, color="skyblue", edgecolor="black")
        plt.xlabel("Counterfactual Method", fontsize=16)
        plt.ylabel(f"Average {metric}", fontsize=16)
        plt.title(f"Average {metric} Ratings per Counterfactual Method", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim(0, 5)  # Ratings range from 1 to 5
        metric_plot_path = os.path.join(output_dir, f"{metric}_ratings.png")
        plt.savefig(metric_plot_path)
        print(f"Bar chart for {metric} saved: {metric_plot_path}")
    else:
        logging.warning(f"No valid ratings available for {metric}. Skipping plot.")
