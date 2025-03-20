# location: carla_dataset_collection/dataset_details.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def process_2_class_dataset(train_csv, test_csv):
    # Check if files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train file not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test file not found: {test_csv}")

    # Read CSV files
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    # Replace 'Label' with the actual column name in your CSV
    label_column = 'label'  # Update this to match your CSV structure

    # Count STOP and GO for train dataset
    train_counts = train_data[label_column].value_counts()

    # Count STOP and GO for test dataset
    test_counts = test_data[label_column].value_counts()

    # Combine the results into a summary table
    summary = pd.DataFrame({
        'Category': ['STOP', 'GO'],
        'Train Count': [train_counts.get('STOP', 0), train_counts.get('GO', 0)],
        'Test Count': [test_counts.get('STOP', 0), test_counts.get('GO', 0)],
    })

    # Add total counts
    summary.loc[len(summary)] = ['Total', summary['Train Count'].sum(), summary['Test Count'].sum()]

    return summary

def process_4_class_dataset(train_csv, test_csv):
    # Check if files exist
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train file not found: {train_csv}")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test file not found: {test_csv}")

    # Read CSV files
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    # Replace 'Label' with the actual column name in your CSV
    label_column = 'label'  # Update this to match your CSV structure

    # Count STOP, GO, RIGHT, and LEFT for train dataset
    train_counts = train_data[label_column].value_counts()

    # Count STOP, GO, RIGHT, and LEFT for test dataset
    test_counts = test_data[label_column].value_counts()

    # Combine the results into a summary table
    summary = pd.DataFrame({
        'Category': ['STOP', 'GO', 'RIGHT', 'LEFT'],
        'Train Count': [train_counts.get('STOP', 0), train_counts.get('GO', 0), train_counts.get('RIGHT', 0), train_counts.get('LEFT', 0)],
        'Test Count': [test_counts.get('STOP', 0), test_counts.get('GO', 0), test_counts.get('RIGHT', 0), test_counts.get('LEFT', 0)],
    })

    # Add total counts
    summary.loc[len(summary)] = ['Total', summary['Train Count'].sum(), summary['Test Count'].sum()]

    return summary

# Pie chart visualization with count labels
def plot_summary_with_pie(summary, dataset_type):
    # Separate Train and Test counts
    categories = summary['Category'][:-1]  # Exclude 'Total' from the graph
    train_counts = summary['Train Count'][:-1]
    test_counts = summary['Test Count'][:-1]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot pie chart for Train dataset
    ax1.pie(train_counts, labels=categories, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.set_title('Train Dataset')

    # Plot pie chart for Test dataset
    ax2.pie(test_counts, labels=categories, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax2.set_title('Test Dataset')

    # Add a main title for the figure
    plt.suptitle(f'{dataset_type} Distribution in Train and Test Datasets')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(f"plots/dataset_images/{dataset_type.lower()}_distribution_pie_chart.png")
    plt.close()

# Table visualization using Seaborn
def plot_table(summary, dataset_type):
    plt.figure(figsize=(6, 2))
    plt.axis('off')  # Turn off the axes
    sns.heatmap(
        summary.set_index('Category'),
        annot=True,
        fmt='g',
        cmap='Blues',
        cbar=False,
        linewidths=0.5
    )
    plt.title(f"{dataset_type} Dataset Details Table")
    plt.tight_layout()

    # Save and show the table plot
    plt.savefig(f"plots/dataset_images/{dataset_type.lower()}_dataset_details_table.png")
    plt.close()

# File paths for 2-class dataset
train_csv_2class = "dataset/town7_dataset/train/labeled_train_data_log.csv"
test_csv_2class = "dataset/town7_dataset/test/labeled_test_data_log.csv"

# File paths for 4-class dataset
train_csv_4class = "dataset/town7_dataset/train/labeled_train_4_class_data_log.csv"
test_csv_4class = "dataset/town7_dataset/test/labeled_test_4_class_data_log.csv"

# Process 2-class dataset
summary_2class = process_2_class_dataset(train_csv_2class, test_csv_2class)
print("2-class dataset summary:")
print(summary_2class)
summary_2class.to_csv("plots/dataset_images/2class_dataset_summary.csv", index=False)
plot_summary_with_pie(summary_2class, "2-class")
plot_table(summary_2class, "2-class")

# Process 4-class dataset
summary_4class = process_4_class_dataset(train_csv_4class, test_csv_4class)
print("\n4-class dataset summary:")
print(summary_4class)
summary_4class.to_csv("plots/dataset_images/4class_dataset_summary.csv", index=False)
plot_summary_with_pie(summary_4class, "4-class")
plot_table(summary_4class, "4-class")