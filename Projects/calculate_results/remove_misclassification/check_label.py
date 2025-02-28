import os
import pandas as pd

# Define the paths for all method result CSVs (both 4-class and 2-class)
methods_results_4_class = {
    "grid_based": "results/masking/grid_based/grid_based_masking_4_classes_results.csv",
    "object_detection": "results/masking/object_detection/object_detection_masking_4_classes_results.csv",
    "lime_on_image": "results/masking/lime_on_images/lime_on_image_masking_4_classes_results.csv",
    "lime_on_latent": "results/masking/lime_on_latent/lime_on_latent_masking_4_classes_results.csv"
}

methods_results_2_class = {
    "grid_based": "results/masking/grid_based/grid_based_masking_2_classes_results.csv",
    "object_detection": "results/masking/object_detection/object_detection_masking_2_classes_results.csv",
    "lime_on_image": "results/masking/lime_on_images/lime_on_image_masking_2_classes_results.csv",
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

# Run for 4-class
find_misclassified_images(methods_results_4_class, "results/classification/misclassified_images_4_classes.csv")

# Run for 2-class
find_misclassified_images(methods_results_2_class, "results/classification/misclassified_images_2_classes.csv")
