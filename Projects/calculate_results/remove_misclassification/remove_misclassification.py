import os
import pandas as pd

# Define the paths for all method result CSVs (both 4-class and 2-class)
methods_results_4_class = {
    "grid_based": "results/masking/grid_based/grid_based_masking_4_classes_results.csv",
    "object_detection": "results/masking/object_detection/object_detection_masking_4_classes_results.csv",
    "lime_on_images": "results/masking/lime_on_images/lime_on_image_masking_4_classes_results.csv",
    "lime_on_latent": "results/masking/lime_on_latent/lime_on_latent_masking_4_classes_results.csv"
}

methods_results_2_class = {
    "grid_based": "results/masking/grid_based/grid_based_masking_2_classes_results.csv",
    "object_detection": "results/masking/object_detection/object_detection_masking_2_classes_results.csv",
    "lime_on_images": "results/masking/lime_on_images/lime_on_image_masking_2_classes_results.csv",
    "lime_on_latent": "results/masking/lime_on_latent/lime_on_latent_masking_2_classes_results.csv"
}

def find_misclassified_images(methods_results, output_file):
    """
    Identifies misclassified images where `Prediction (Before Masking)` differs across methods.
    Saves the list of misclassified images to a CSV file.
    """
    dfs = {}
    for method, file_path in methods_results.items():
        if os.path.exists(file_path):
            dfs[method] = pd.read_csv(file_path)
            print(f"Loaded {method} results from {file_path}")
        else:
            print(f"Warning: {file_path} not found. Skipping {method}.")

    # Merge DataFrames based on "Image File" and "Prediction (Before Masking)"
    merged_df = None
    for method, df in dfs.items():
        df = df[["Image File", "Prediction (Before Masking)"]]
        df = df.rename(columns={"Prediction (Before Masking)": f"Prediction ({method})"})
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="Image File", how="outer")

    # Identify images with inconsistent initial predictions
    misclassified_images = merged_df[
        merged_df.iloc[:, 1:].nunique(axis=1) > 1  # Count unique predictions across methods
    ]

    # Save misclassified images to CSV
    misclassified_images.to_csv(output_file, index=False)

    # Print summary
    total_images = len(merged_df)
    misclassified_count = len(misclassified_images)

    print(f"Total Images Processed: {total_images}")
    print(f"Misclassified Images: {misclassified_count}")
    print(f"List of misclassified images saved to {output_file}")

    return misclassified_images

# Run for 4-class
misclassified_4c = find_misclassified_images(methods_results_4_class, "results/classification/misclassified_images_4_classes.csv")

# Run for 2-class
misclassified_2c = find_misclassified_images(methods_results_2_class, "results/classification/misclassified_images_2_classes.csv")

# Get misclassified images separately for 4-class and 2-class
misclassified_images_4c = set(misclassified_4c["Image File"])
misclassified_images_2c = set(misclassified_2c["Image File"])

print(f"Removing {len(misclassified_images_4c)} misclassified images from 4-class evaluation.")
print(f"Removing {len(misclassified_images_2c)} misclassified images from 2-class evaluation.")

# Paths to the original result files
result_files_4c = list(methods_results_4_class.values())
result_files_2c = list(methods_results_2_class.values())

# Function to clean result files by removing misclassified images
def remove_misclassified_images(result_files, misclassified_images):
    for file in result_files:
        df = pd.read_csv(file)
        initial_count = len(df)
        
        # Remove misclassified images
        df = df[~df["Image File"].isin(misclassified_images)]
        
        # Save the cleaned file
        df.to_csv(file, index=False)
        print(f"Updated {file}: {initial_count} → {len(df)} rows after removal.")

# Remove misclassified images **only from respective methods**
remove_misclassified_images(result_files_4c, misclassified_images_4c)
remove_misclassified_images(result_files_2c, misclassified_images_2c)

print(" Misclassified images removed separately for 4-class and 2-class evaluations.")
