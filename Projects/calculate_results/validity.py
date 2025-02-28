import os
import pandas as pd

# ------------------------------------------------------------------------------
# Process both 2_class and 4_class results
# ------------------------------------------------------------------------------
class_types = ["2_class", "4_class"]

for class_type in class_types:
    # Determine file suffix based on class type
    class_suffix = "2_classes" if class_type == "2_class" else "4_classes"
    
    # ------------------------------------------------------------------------------
    # Define file paths for each method using the new directory structure
    # ------------------------------------------------------------------------------
    METHODS_FILES = {
        "Grid-Based Masking": f"results/masking/grid_based/grid_based_masking_{class_suffix}_results.csv",
        "Object Detection": f"results/masking/object_detection/object_detection_masking_{class_suffix}_results.csv",
        "LIME on Images": f"results/masking/lime_on_images/lime_on_image_masking_{class_suffix}_results.csv",
        "LIME on Latent Features": f"results/masking/lime_on_latent/lime_on_latent_masking_{class_suffix}_results.csv"
    }
    
    # ------------------------------------------------------------------------------
    # Set up output directory for validity check for the current class type
    # ------------------------------------------------------------------------------
    validity_dir = os.path.join("results", "validity_check", class_type)
    os.makedirs(validity_dir, exist_ok=True)
    validity_csv_path = os.path.join(validity_dir, f"validity_summary_{class_type}.csv")
    
    # ------------------------------------------------------------------------------
    # Function: Compute Validity for a method
    # ------------------------------------------------------------------------------
    def calculate_validity(df: pd.DataFrame, method_name: str) -> dict:
        total_counterfactuals = len(df)
        successful_counterfactuals = df[df["Counterfactual Found"] == True].shape[0]
        validity = (successful_counterfactuals / total_counterfactuals * 100) if total_counterfactuals > 0 else 0
        return {
            "Method": method_name,
            "Total Counterfactuals": total_counterfactuals,
            "Successful Counterfactuals": successful_counterfactuals,
            "Validity (%)": round(validity, 2)
        }
    
    # ------------------------------------------------------------------------------
    # Load Data and Compute Validity for Each Method
    # ------------------------------------------------------------------------------
    validity_results = []
    
    for method, filepath in METHODS_FILES.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            validity_results.append(calculate_validity(df, method))
        else:
            print(f"⚠ Warning: {filepath} not found. Skipping {method}.")
    
    # Convert results into a DataFrame and save the CSV
    validity_df = pd.DataFrame(validity_results)
    validity_df.to_csv(validity_csv_path, index=False)
    
    # ------------------------------------------------------------------------------
    # Print Validity Summary
    # ------------------------------------------------------------------------------
    print(f"\nValidity Summary for {class_type}:")
    print(validity_df.to_string(index=False))
    print(f"\nValidity summary saved to: {os.path.abspath(validity_csv_path)}")
