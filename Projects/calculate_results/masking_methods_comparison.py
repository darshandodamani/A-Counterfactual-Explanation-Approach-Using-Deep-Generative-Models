# location: Projects/calculate_results/masking_methods_comparison.py
import os
import sys
import logging
from typing import Set, Dict

import pandas as pd
import matplotlib.pyplot as plt
from venny4py.venny4py import venny4py

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# ------------------------------------------------------------------------------
# Directory & File Paths for CE Results
# ------------------------------------------------------------------------------
methods_results = {
    "2_class": {
        "grid_based": "results/masking/grid_based/grid_based_masking_2_classes_results.csv",
        "object_detection": "results/masking/object_detection/object_detection_masking_2_classes_results.csv",
        "lime_on_image": "results/masking/lime_on_images/lime_on_image_masking_2_classes_results.csv",
        "lime_on_latent": "results/masking/lime_on_latent/lime_on_latent_masking_2_classes_results.csv",
        "lime_on_latent_nun": "results/masking/lime_on_latent/2_class_NUN_results.csv"
    },
    "4_class": {
        "grid_based": "results/masking/grid_based/grid_based_masking_4_classes_results.csv",
        "object_detection": "results/masking/object_detection/object_detection_masking_4_classes_results.csv",
        "lime_on_image": "results/masking/lime_on_images/lime_on_image_masking_4_classes_results.csv",
        "lime_on_latent": "results/masking/lime_on_latent/lime_on_latent_masking_4_classes_results.csv",
        "lime_on_latent_nun": "results/masking/lime_on_latent/4_class_NUN_results.csv"
    }
}

# Output paths for individual method summaries (computed in CE summary)
output_method_summary_paths = {
    "2_class": {
        "grid_based": "results/masking/grid_based/grid_based_2_class_summary.csv",
        "object_detection": "results/masking/object_detection/object_detection_2_class_summary.csv",
        "lime_on_image": "results/masking/lime_on_images/lime_on_image_2_class_summary.csv",
        "lime_on_latent": "results/masking/lime_on_latent/lime_on_latent_2_class_summary.csv",
        "lime_on_latent_nun": "results/masking/lime_on_latent/2_class_NUN_summary.csv"
    },
    "4_class": {
        "grid_based": "results/masking/grid_based/grid_based_4_class_summary.csv",
        "object_detection": "results/masking/object_detection/object_detection_4_class_summary.csv",
        "lime_on_image": "results/masking/lime_on_images/lime_on_image_4_class_summary.csv",
        "lime_on_latent": "results/masking/lime_on_latent/lime_on_latent_4_class_summary.csv",
        "lime_on_latent_nun": "results/masking/lime_on_latent/4_class_NUN_summary.csv"
    }
}

# Output overall summary paths for CE metrics
output_summary = {
    "2_class": "results/masking/summary_2_class_comparison.csv",
    "4_class": "results/masking/summary_4_class_comparison.csv"
}

# Base directory for methods overlap analysis outputs (Venn diagram, bar chart, and comparison table)
BASE_METHOD_COMPARISON_DIR = os.path.join("plots", "venn_diagram")
os.makedirs(BASE_METHOD_COMPARISON_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Compute Counterfactual Explanation (CE) Summary
# ------------------------------------------------------------------------------
def compute_ce_summary(class_type: str) -> None:
    """
    Computes CE Found metrics for each masking method for the specified class type (2_class or 4_class).
    Saves both an overall summary and individual method summaries.
    """
    summary_data = []
    # Define the categories based on class type
    categories = ["STOP", "GO"] if class_type == "2_class" else ["STOP", "GO", "LEFT", "RIGHT"]

    for method, file_path in methods_results[class_type].items():
        if not os.path.exists(file_path):
            logging.warning(f"File {file_path} not found. Skipping method '{method}'.")
            continue
        
        df = pd.read_csv(file_path)
        total_entries = len(df)
        total_ce_found, total_ce_not_found = 0, 0
        method_summary = {"Method": method, "Total": total_entries}

        for category in categories:
            category_cases = df[df["Prediction (Before Masking)"] == category]
            ce_found = len(category_cases[category_cases["Counterfactual Found"] == True])
            ce_not_found = len(category_cases[category_cases["Counterfactual Found"] == False])
            ce_found_pct = (ce_found / len(category_cases) * 100) if len(category_cases) > 0 else 0
            ce_not_found_pct = (ce_not_found / len(category_cases) * 100) if len(category_cases) > 0 else 0

            method_summary[f"Total {category}"] = len(category_cases)
            method_summary[f"{category} CE Found"] = ce_found
            method_summary[f"{category} CE Found (%)"] = f"{ce_found_pct:.2f}%"
            method_summary[f"{category} CE Not Found"] = ce_not_found
            method_summary[f"{category} CE Not Found (%)"] = f"{ce_not_found_pct:.2f}%"

            total_ce_found += ce_found
            total_ce_not_found += ce_not_found

        total_ce_found_pct = (total_ce_found / total_entries * 100) if total_entries > 0 else 0
        total_ce_not_found_pct = (total_ce_not_found / total_entries * 100) if total_entries > 0 else 0
        total_time_taken_sec = round(df["Time Taken (s)"].sum(), 2)
        total_time_taken_min = round(total_time_taken_sec / 60, 2)

        method_summary["Total CE Found"] = total_ce_found
        method_summary["Total CE Found (%)"] = f"{total_ce_found_pct:.2f}%"
        method_summary["Total CE Not Found"] = total_ce_not_found
        method_summary["Total CE Not Found (%)"] = f"{total_ce_not_found_pct:.2f}%"
        method_summary["Total Time Taken (s)"] = total_time_taken_sec
        method_summary["Total Time Taken (min)"] = total_time_taken_min

        summary_data.append(method_summary)
        
        # Save individual method summary
        method_summary_path = output_method_summary_paths[class_type][method]
        os.makedirs(os.path.dirname(method_summary_path), exist_ok=True)
        pd.DataFrame([method_summary]).to_csv(method_summary_path, index=False)
        logging.info(f"Saved {method} summary to {os.path.abspath(method_summary_path)}")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_summary[class_type], index=False)
    logging.info(f"CE summary for {class_type.replace('_', '-')} saved to {os.path.abspath(output_summary[class_type])}")

# ------------------------------------------------------------------------------
# Data Loading and Overlap Analysis Functions
# ------------------------------------------------------------------------------
def load_dataframe(csv_file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded {csv_file} with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading {csv_file}: {e}")
        sys.exit(1)

def get_counterfactual_set(df: pd.DataFrame) -> Set[str]:
    """Returns a set of image file names for which a counterfactual was found."""
    return set(df[df["Counterfactual Found"] == True]["Image File"])

def generate_comparison_table(all_images: Set[str],
                              grid_ce: Set[str],
                              lime_latent_ce: Set[str],
                              lime_image_ce: Set[str],
                              object_detection_ce: Set[str]) -> pd.DataFrame:
    """Generates a comparison table indicating for each image whether a counterfactual was found by each method."""
    table = []
    for image in sorted(all_images):
        table.append({
            "Image File": image,
            "Grid-Based": "True" if image in grid_ce else "False",
            "LIME on Latent Features": "True" if image in lime_latent_ce else "False",
            "LIME on Images": "True" if image in lime_image_ce else "False",
            "Object Detection": "True" if image in object_detection_ce else "False",
        })
    return pd.DataFrame(table)

def generate_venn_diagram(sets: Dict[str, Set[str]], venn_path: str) -> None:
    """Generates and saves a Venn diagram for the provided method sets."""
    plt.figure(figsize=(12, 8))
    venny4py(sets=sets)
    plt.title("Venn Diagram of Counterfactual Explanation Coverage", fontsize=16)
    plt.savefig(venn_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Venn diagram saved as '{os.path.abspath(venn_path)}'")

def generate_bar_chart(grid_ce: Set[str], 
                       lime_latent_ce: Set[str], 
                       lime_image_ce: Set[str],
                       lime_latent_nun_ce: Set[str],
                       bar_chart_path: str) -> None:
    """Generates and saves a bar chart showing the number of images explained by each method."""
    methods = ["Grid-Based", "LIME on Latent", "LIME on Images", "LIME on Latent NUN"]
    counts = [len(grid_ce), len(lime_latent_ce), len(lime_image_ce), len(lime_latent_nun_ce)]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, counts, color=['blue', 'orange', 'green', 'purple'])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, str(yval),
                 ha='center', fontsize=12, fontweight='bold')
    plt.xlabel("Masking Methods", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Images Explained", fontsize=12, fontweight='bold')
    plt.title("Counterfactual Explanations per Method", fontsize=14, fontweight='bold')
    plt.xticks(rotation=25)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(bar_chart_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Bar chart saved as '{os.path.abspath(bar_chart_path)}'")

def generate_overlap_summary(all_images: Set[str], 
                             grid_ce: Set[str], 
                             lime_latent_ce: Set[str], 
                             lime_image_ce: Set[str], 
                             lime_latent_nun_ce: Set[str],
                             summary_path: str) -> None:
    """Generates a summary of the overlap between the methods."""
    total_images = len(all_images)
    
    categories = [
        "Only Grid-Based Masking",
        "Only LIME on Latent Features",
        "Only LIME on Images",
        "Only LIME on Latent NUN",
        "Grid-Based + LIME on Latent Features",
        "Grid-Based + LIME on Images",
        "Grid-Based + LIME on Latent NUN",
        "LIME on Latent Features + LIME on Images",
        "LIME on Latent Features + LIME on Latent NUN",
        "LIME on Images + LIME on Latent NUN",
        "Grid-Based + LIME on Latent + LIME on Images",
        "Grid-Based + LIME on Latent + LIME on Latent NUN",
        "Grid-Based + LIME on Images + LIME on Latent NUN",
        "LIME on Latent + LIME on Images + LIME on Latent NUN",
        "All Four Methods",
        "Not Explained by Any Method"
    ]
    
    counts = []
    
    # Calculate counts for each category
    counts.append(len(grid_ce - lime_latent_ce - lime_image_ce - lime_latent_nun_ce))
    counts.append(len(lime_latent_ce - grid_ce - lime_image_ce - lime_latent_nun_ce))
    counts.append(len(lime_image_ce - grid_ce - lime_latent_ce - lime_latent_nun_ce))
    counts.append(len(lime_latent_nun_ce - grid_ce - lime_latent_ce - lime_image_ce))
    counts.append(len(grid_ce.intersection(lime_latent_ce) - lime_image_ce - lime_latent_nun_ce))
    counts.append(len(grid_ce.intersection(lime_image_ce) - lime_latent_ce - lime_latent_nun_ce))
    counts.append(len(grid_ce.intersection(lime_latent_nun_ce) - lime_latent_ce - lime_image_ce))
    counts.append(len(lime_latent_ce.intersection(lime_image_ce) - grid_ce - lime_latent_nun_ce))
    counts.append(len(lime_latent_ce.intersection(lime_latent_nun_ce) - grid_ce - lime_image_ce))
    counts.append(len(lime_image_ce.intersection(lime_latent_nun_ce) - grid_ce - lime_latent_ce))
    counts.append(len(grid_ce.intersection(lime_latent_ce).intersection(lime_image_ce) - lime_latent_nun_ce))
    counts.append(len(grid_ce.intersection(lime_latent_ce).intersection(lime_latent_nun_ce) - lime_image_ce))
    counts.append(len(grid_ce.intersection(lime_image_ce).intersection(lime_latent_nun_ce) - lime_latent_ce))
    counts.append(len(lime_latent_ce.intersection(lime_image_ce).intersection(lime_latent_nun_ce) - grid_ce))
    counts.append(len(grid_ce.intersection(lime_latent_ce).intersection(lime_image_ce).intersection(lime_latent_nun_ce)))
    counts.append(len(all_images - grid_ce - lime_latent_ce - lime_image_ce - lime_latent_nun_ce))
    
    # Create DataFrame
    summary_data = {
        "Category": categories,
        "Number of Images": counts,
        "Percentage (%)": [f"{(count / total_images * 100):.2f}" for count in counts]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add total images row
    total_row = pd.DataFrame({
        "Category": ["Total Images"],
        "Number of Images": [total_images],
        "Percentage (%)": ["100"]
    })
    
    summary_df = pd.concat([total_row, summary_df], ignore_index=True)
    
    # Save summary to CSV
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Overlap summary saved as '{os.path.abspath(summary_path)}'")

# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Main function that:
    1. Computes the counterfactual explanation summaries.
    2. Loads the result CSVs for each class type.
    3. Generates the methods overlap comparison table, Venn diagram, bar chart, and overlap summary.
    4. Additionally, for each class type, saves a CSV file containing the intersection of images
       explained by Grid-Based Masking, LIME on Latent Features, and LIME on Images along with
       their predictions.
    """
    # Process both "2_class" and "4_class" by default
    class_types = list(methods_results.keys())

    for class_type in class_types:
        logging.info(f"Running analysis for {class_type}...")
        # Compute CE summary for current class type
        compute_ce_summary(class_type)

        # File paths for current class type
        current_files = methods_results[class_type]
        grid_csv = current_files["grid_based"]
        lime_image_csv = current_files["lime_on_image"]
        lime_latent_csv = current_files["lime_on_latent"]
        lime_latent_nun_csv = current_files["lime_on_latent_nun"]

        # Load DataFrames for each method
        grid_df = load_dataframe(grid_csv)
        lime_latent_df = load_dataframe(lime_latent_csv)
        lime_image_df = load_dataframe(lime_image_csv)
        lime_latent_nun_df = load_dataframe(lime_latent_nun_csv)

        # Generate sets of images where a counterfactual was found
        grid_ce = get_counterfactual_set(grid_df)
        lime_latent_ce = get_counterfactual_set(lime_latent_df)
        lime_image_ce = get_counterfactual_set(lime_image_df)
        lime_latent_nun_ce = get_counterfactual_set(lime_latent_nun_df)

        # Combine all unique image file names across methods
        all_images = (set(grid_df["Image File"]) |
                      set(lime_latent_df["Image File"]) |
                      set(lime_image_df["Image File"]) |
                      set(lime_latent_nun_df["Image File"]))

        # Define output paths specific to the class type
        table_output_path = os.path.join(BASE_METHOD_COMPARISON_DIR, f"method_comparison_table_{class_type}.csv")
        venn_diagram_path = os.path.join(BASE_METHOD_COMPARISON_DIR, f"venn_{class_type}.png")
        bar_chart_path = os.path.join(BASE_METHOD_COMPARISON_DIR, f"bar_chart_explanations_{class_type}.png")
        summary_path = os.path.join(BASE_METHOD_COMPARISON_DIR, f"overlap_summary_{class_type}.csv")

        # Generate and save the comparison table
        comparison_table = generate_comparison_table(all_images, grid_ce, lime_latent_ce, lime_image_ce, lime_latent_nun_ce)
        comparison_table.to_csv(table_output_path, index=False)
        logging.info(f"Comparison table saved as '{os.path.abspath(table_output_path)}'")

        # Generate and save the Venn diagram, bar chart, and overlap summary
        venn_sets = {
            "Grid-Based Masking": grid_ce,
            "LIME on Latent Features": lime_latent_ce,
            "LIME on Images": lime_image_ce,
            "LIME on Latent NUN": lime_latent_nun_ce
        }
        generate_venn_diagram(venn_sets, venn_diagram_path)
        generate_bar_chart(grid_ce, lime_latent_ce, lime_image_ce, lime_latent_nun_ce, bar_chart_path)
        generate_overlap_summary(all_images, grid_ce, lime_latent_ce, lime_image_ce, lime_latent_nun_ce, summary_path)

        # --------------------------------------------------------------------------
        # Save CSV for Intersection of Grid-Based, LIME on Latent, and LIME on Images
        # --------------------------------------------------------------------------
        # Compute intersection of images explained by Grid-Based, LIME on Latent, and LIME on Images
        intersection_images = grid_ce.intersection(lime_latent_ce).intersection(lime_image_ce)
        if intersection_images:
            # For these images, we extract the predictions from the grid-based results (assumed to be representative)
            intersection_df = grid_df[grid_df["Image File"].isin(intersection_images)][["Image File", "Prediction (Before Masking)", "Prediction (After Masking)"]]
            intersection_csv_path = os.path.join(BASE_METHOD_COMPARISON_DIR, f"intersection_combined_{class_type}.csv")
            intersection_df.to_csv(intersection_csv_path, index=False)
            logging.info(f"Intersection CSV saved as '{os.path.abspath(intersection_csv_path)}'")
        else:
            logging.info("No intersection found for Grid-Based, LIME on Latent, and LIME on Images.")

    print("\nAll CE summaries and visualizations have been successfully generated.")

if __name__ == "__main__":
    main()