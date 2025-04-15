1. **After Cloning the git repository follow the next steps**
2. **Create Virtual Environment**: We recommend creating a Python virtual environment for this project:

   ```bash
   python -m venv venv_carla
   ```

   Activate the virtual environment:

   ```bash
   source venv_carla/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies**: Install the required dependencies using:

   ```bash
   pip install -r requirements_carla.txt
   ```
4. **Download CARLA Server**: Download **CARLA server (0.9.15)** and **additional maps** from the [official CARLA repository](https://github.com/carla-simulator/carla/releases).
5. **Start CARLA Server**: Make sure to start the CARLA server before running the client.

   ```bash
   ./CarlaUE4.sh
   ```

# Dataset Collection

Currently, the dataset is collected from **Town07** and **Town06** environments in CARLA. This dataset will be used for training the deep generative model and generating counterfactual explanations.

### Running Dataset Collection

To collect the dataset, use the following command:

```bash
python carla_dataset_collection/collect_images.py --output_dir dataset/town7_dataset --town_name Town07 --image_size 160 80
```

- **Parameters**:
  - `--output_dir`: Directory where the collected dataset will be stored.
  - `--town_name`: Specify the CARLA town (e.g., Town07, Town06).
  - `--image_size`: Specify the width and height of the collected images.

# Labeling and Splitting the Dataset

Once the dataset is collected, the next steps are **labeling** and **splitting** the dataset for training and testing purposes.

### Labeling the Dataset

The dataset is labeled based on the throttle and brake values collected from the vehicle in CARLA. Images are labeled as `STOP` or `GO`  for binary class labelling and `STOP`, `GO`, `RIGHT` or `LEFT`  for multi-class labeling based on thresholds for brake and throttle:

These are calculated using the threshold values for the data collected. Calculate the threshold and define the threshold depending on which the labelling will be done.

- **STOP**: If `brake > 0.5` or `throttle < 0.2`,
- **GO**: Otherwise.

- **STOP**: If `brake > 0.5` or `throttle < 0.2`,
- **GO**: If `steering > 0.2` or `brake < 0.5`,
- **RIGHT**: If `steering > -0.1` or `brake < 0.5`,
- **LEFT**: If `throttle > 0.2` or `|steering| < 0.1`,

To label the dataset, use the following command:

```bash
python carla_dataset_collection/label_dataset.py
```

### Splitting the Dataset

To prepare the dataset for training, it needs to be split into **training** and **testing** subsets while maintaining balanced classes (`STOP` and `GO`). The splitting can be done using the command below:

```bash
python carla_dataset_collection/split_dataset.py --data_path <path_to_dataset> --train_ratio 0.8
```

- Example:
  ```bash
  python carla_dataset_collection/split_dataset.py --data_path ../dataset/town7_dataset --train_ratio 0.8
  ```
- **Parameters**:
  - `--train_ratio`: Ratio of the data to use for training (default: 0.8).