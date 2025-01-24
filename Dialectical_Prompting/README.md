# Dialectical Prompting

This repository contains the implementation of the **Dialectical Prompting** approach for improving text classification tasks. Dialectical Prompting leverages the power of large language models (LLMs) to generate high-quality explanations, which are then used to guide the training of smaller, more specialized models. This method enhances both the accuracy and interpretability of text classification models by incorporating explanations into the training process.


## 0. Repository Structure

- **`configs/`**: Contains configuration files for running experiments and training models.
- **`core/`**: Core modules and functions used across different parts of the project.
- **`data/`**: Contains datasets used for training, validation, and testing.
- **`EQ-Evaluating-Input/`**: Input data and configuration files for evaluating explanation quality.
- **`EQ-Evaluating-Output/`**: Outputs from the evaluation of explanation quality.
- **`human_experts_test/`**: Data and scripts related to testing explanation quality with human experts.
- **`outputs/`**: Directory where output files from model runs and experiments are saved.
- **`prompts/`**: Contains the prompts used for generating explanations with LLMs.
- **`AV_EQ_score_for_models.py`**: Script for calculating the Explanation Quality (EQ) scores for the models.
- **`Evaluating_explanation_quality.py`**: Script to evaluate the quality of explanations generated by the models.
- **`requirements.txt`**: Lists the required Python packages and dependencies to set up the environment.
- **`README.md`**: This file, providing an overview of the repository, its structure, and usage instructions.
- **`main_experiment.py`**: The main script to run experiments for Dialetical Prompting.

## 1. Python environment setup
   
- **`Prepare for the conda environment.`**:
   * `conda create -n dialprompt python=3.8.18`
   * `conda activate dialprompt`
- **`Install Python dependencies and process the data.`**:
   * `chmod +x setup.sh`
   * `./setup.sh`

## 2. Run experiments for Dialectical Prompting
- **`Configure`**:
   * the llms api key setting in [configs/llm.yaml](configs/llm.yaml).
   * the experiments in [configs/experiment.yaml](configs/experiment.yaml).
- **`Look into [main_experiments.py](main_experiments.py) to see how to run experiments.`**:
