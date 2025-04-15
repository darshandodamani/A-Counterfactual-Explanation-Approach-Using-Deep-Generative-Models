import pandas as pd
import matplotlib.pyplot as plt
import os

def label_dataset(input_csv, output_csv, plot_path):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Define realistic thresholds defined after calculations
    stop_brake_threshold = 0.5  # Brake > 0.5
    stop_throttle_threshold = 0.2  # Throttle < 0.2
    go_throttle_threshold = 0.5  # Throttle > 0.5
    go_steering_threshold = 0.1  # |Steering| < 0.1
    turn_steering_threshold = 0.1  # Steering > 0.1 or < -0.1
    turn_brake_threshold = 0.5  # Brake < 0.5

    print("Adjusted Thresholds:")
    print(f"STOP: brake > {stop_brake_threshold}, throttle < {stop_throttle_threshold}")
    print(f"GO: throttle > {go_throttle_threshold}, abs(steering) < {go_steering_threshold}")
    print(f"RIGHT: steering > {turn_steering_threshold}, brake < {turn_brake_threshold}")
    print(f"LEFT: steering < {-turn_steering_threshold}, brake < {turn_brake_threshold}")

    # Label data based on thresholds
    def classify_row(row):
        if row['brake'] > stop_brake_threshold and row['throttle'] < stop_throttle_threshold:
            return 'STOP'
        elif row['throttle'] > go_throttle_threshold and abs(row['steering']) < go_steering_threshold:
            return 'GO'
        elif row['steering'] > turn_steering_threshold and row['brake'] < turn_brake_threshold:
            return 'RIGHT'
        elif row['steering'] < -turn_steering_threshold and row['brake'] < turn_brake_threshold:
            return 'LEFT'
        else:
            return 'UNKNOWN'

    df['label'] = df.apply(classify_row, axis=1)

    # Debugging: Print counts for each condition
    print(f"Rows classified as STOP: {len(df[df['label'] == 'STOP'])}")
    print(f"Rows classified as GO: {len(df[df['label'] == 'GO'])}")
    print(f"Rows classified as RIGHT: {len(df[df['label'] == 'RIGHT'])}")
    print(f"Rows classified as LEFT: {len(df[df['label'] == 'LEFT'])}")

    # Filter out UNKNOWN labels
    df = df[df['label'] != 'UNKNOWN']

    # Save the labeled dataset
    df.to_csv(output_csv, index=False)
    print(f"Labeled dataset saved to {output_csv}")

    # Plot label distribution
    label_counts = df['label'].value_counts()
    print("Label distribution:")
    print(label_counts)

    plt.figure(figsize=(8, 8))
    label_counts.plot.pie(autopct='%1.1f%%', startangle=90, explode=[0.1] * len(label_counts))
    plt.title(f'Label Distribution')
    plt.ylabel('')
    plt.savefig(plot_path)
    plt.close()
    print(f"Label distribution plot saved to {plot_path}")

# Train dataset processing
label_dataset(
    input_csv=os.path.join("dataset/town7_dataset/train", "train_data_log.csv"),
    output_csv=os.path.join("dataset/town7_dataset/train", "labeled_train_4_class_data_log.csv"),
    plot_path=os.path.join("plots/dataset_images_for_4_classes", "train_label_distribution.png")
)

# Test dataset processing
label_dataset(
    input_csv=os.path.join("dataset/town7_dataset/test", "test_data_log.csv"),
    output_csv=os.path.join("dataset/town7_dataset/test", "labeled_test_4_class_data_log.csv"),
    plot_path=os.path.join("plots/dataset_images_for_4_classes", "test_label_distribution.png")
)
