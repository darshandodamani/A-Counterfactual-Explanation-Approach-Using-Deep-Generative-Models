import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load NUN results CSV
nun_csv_path = "latent_vectors/nun_values.csv"
nun_df = pd.read_csv(nun_csv_path)

# Check dataset summary
print("\nDataset Overview:")
print(nun_df.head())

# Ensure Distance column is numeric
nun_df["Distance"] = pd.to_numeric(nun_df["Distance"], errors='coerce')

# ----------------------------------------------------------------------------
# 1️⃣ Histogram of NUN Distances
# ----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.histplot(nun_df["Distance"], bins=30, kde=True)
plt.xlabel("Distance in Latent Space")
plt.ylabel("Number of Images")
plt.title("Distribution of NUN Distances")
plt.grid(True)
plt.savefig("latent_vectors/nun_distance_histogram.png")

# ----------------------------------------------------------------------------
# 2️⃣ Identify Images Without a NUN
# ----------------------------------------------------------------------------
no_nun_df = nun_df[nun_df["NUN Image"] == "None"]
print(f"\nImages with No NUN: {len(no_nun_df)}")
if not no_nun_df.empty:
    print(no_nun_df[["Image File", "Query Label"]])

# ----------------------------------------------------------------------------
# 3️⃣ Compare Latent Feature Differences Between Query and NUN
# ----------------------------------------------------------------------------
# Select sample images for analysis
sample_images = nun_df.sample(5, random_state=42)  # Random 5 samples

# Compute Absolute Differences Between Query and NUN Latents
feature_columns = [col for col in nun_df.columns if col.startswith("latent_")]
latent_differences = []

for _, row in sample_images.iterrows():
    if row["NUN Image"] != "None":
        query_latent = np.array(row[feature_columns], dtype=float)
        nun_latent = np.array(nun_df[nun_df["Image File"] == row["NUN Image"]][feature_columns].values[0], dtype=float)
        diff = np.abs(query_latent - nun_latent)
        latent_differences.append(diff)

# Convert to DataFrame
latent_diff_df = pd.DataFrame(latent_differences, columns=feature_columns)

# Plot Latent Feature Differences
plt.figure(figsize=(28, 14))
sns.boxplot(data=latent_diff_df)
plt.xlabel("Latent Features")
plt.ylabel("Absolute Difference")
plt.title("Feature Importance: Latent Differences Between Query and NUN")
plt.xticks(rotation=90)
plt.savefig("latent_vectors/latent_diff_boxplot.png", dpi=600)

# ----------------------------------------------------------------------------
# 4️⃣ Feature Contribution Analysis (Top Changed Latent Dimensions)
# ----------------------------------------------------------------------------
# Compute Mean Absolute Change Per Feature
feature_importance = latent_diff_df.mean().sort_values(ascending=False)

# Plot Top 20 Changed Features
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.index[:20], y=feature_importance.values[:20], hue=feature_importance.index[:20], palette="viridis", legend=False)

plt.xlabel("Latent Feature Index")
plt.ylabel("Mean Absolute Change")
plt.title("Top 20 Latent Features with Highest Change Between Query & NUN")
plt.xticks(rotation=90)
plt.savefig("latent_vectors/feature_importance.png", bbox_inches="tight")