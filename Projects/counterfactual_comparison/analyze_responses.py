import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================
# Step 0: Load and Clean Data
# ============================
file_path = "Projects/counterfactual_comparison/responses.csv"
df = pd.read_csv(file_path)

# Strip extra whitespace from column names
df.columns = df.columns.str.strip()

# Inspect dataset
print("Dataset Overview:\n", df.head())
print("Detected columns:", df.columns.tolist())

# ============================
# Step 1: Rename Columns (Optional)
# ============================
# Use this mapping if you want to show alternative method names for visualization.
# If your CSV already contains the desired column names, you can skip this step.
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
# Step 6: Bar Plot Using Pivoted Data
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
plt.figure(figsize=(8, 5))
sns.heatmap(df_pivot, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("User Evaluation Heatmap")
plt.xlabel("Method")
plt.ylabel("Evaluation Criterion")
heatmap_path = os.path.join(output_dir, "heatmap_user_evaluations.png")
plt.savefig(heatmap_path)
print(f"Heatmap saved: {heatmap_path}")

# ============================
# Step 8: Compute Average Ratings from Wide-Format Data and Plot Individual Bar Charts
# ============================
metrics = ["Interpretability", "Plausibility", "VisualCoherence"]
methods = [1, 2, 3]
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
    plt.bar([f"Method {m}" for m in methods], avg_ratings[metric], color="skyblue", edgecolor="black")
    plt.xlabel("Counterfactual Method")
    plt.ylabel(f"Average {metric}")
    plt.title(f"Average {metric} Ratings per Counterfactual Method")
    plt.ylim(0, 5)
    metric_plot_path = os.path.join(output_dir, f"{metric}_ratings.png")
    plt.savefig(metric_plot_path)
    print(f"Bar chart for {metric} saved: {metric_plot_path}")
