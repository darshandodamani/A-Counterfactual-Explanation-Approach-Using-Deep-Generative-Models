# A Counterfactual Explanation Approach Using Deep Generative Models

This repository contains the implementation of the research presented in my thesis, which focuses on improving the **explainability** of autonomous driving decisions through **counterfactual explanations**. By leveraging **Deep Generative Models**, this work aims to provide clear insights into the decision-making processes of autonomous vehicles, especially in critical driving scenarios.

# Overview

Autonomous driving systems make complex decisions based on various inputs from their environment. While these decisions are often accurate, there is a growing need to **understand why** a particular action was taken or why an alternative action wasn't chosen. To address this, we use **counterfactual explanations**, which help in determining what minimal changes in the input would alter the model's decision.

# About the Project

This work aims to develop a solution for autonomous driving that can explain the decision-making process using counterfactual explanations. The project involves the following components:

- **CARLA Environment Setup**: Setting up the CARLA simulator to create a realistic driving environment.
- **Variational Autoencoder (VAE)**: A deep generative model used to encode high-dimensional driving data into a latent space, enabling counterfactual analysis.
- **Counterfactual Explanation Generation**: Using feature masking, inpainting, and reconstruction to generate alternative scenarios and understand the model's decisions.

We have used **CARLA (version 0.9.15)** as our urban driving simulator to collect datasets for training and generating counterfactual explanations.

# Prerequisites

- **CARLA Version**: `0.9.15` (Urban Simulator)
- **Python Version**: `3.7` for (CARLA), and `3.11`(VAE) is recommended for compatibility.
- **Additional Maps**: We have focused on **Town07** and **Town06**. Please download the additional maps alongside the CARLA server and copy them into the main CARLA directory to ensure seamless operation.

# Setting Up the Project

**Clone the Repository**: Clone this repository to get started.

### Download the Models from Hugging Face

The trained models are saved in the Hugging Face. To run and test the project.

Download the models from [Hugging Face](https://huggingface.co/darshandodamani/VAE_models/tree/main) and place them in the `model/epochs_500_latent_128_town_7/` directory.

The models are located in the `model/epochs_500_latent_128_town_7/` directory on Hugging Face.


## For Human-centric Evaluations Please run the application

### Running the Web Application

Follow these steps to run the web application locally:

```bash
uvicorn Projects.counterfactual_comparison.app:app --reload
```

This runs the app in development mode with auto-reload enabled. The app will be accessible at http://127.0.0.1:8000.

#### Access the App in Your Browser

- Open your browser and go to http://127.0.0.1:8000/ to view the welcome page.
- Click the Start Evaluation button to begin the counterfactual explanation evaluation.

# Built With

- **Python** - Programming language
- **PyTorch** - Open source machine learning framework
- **CARLA** - An urban driving simulator
- **TensorBoard** - Visualization toolkit
