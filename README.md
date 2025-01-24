
# Explainable Text Classification with LLMs: Enhancing Performance through Dialectical Prompting and Explanation-Guided Training

This repository contains implementations for two complementary approaches aimed at enhancing text classification tasks using Large Language Models (LLMs):

1. **Dialectical Prompting**: A novel method that enhances model performance by simulating a dialectical process where multiple perspectives or explanations are generated for a given input. This approach helps in refining the model’s understanding and improves both accuracy and interpretability.

2. **Explanation Guided Training**: This method focuses on using explanations generated during the training process to guide the training of smaller models. The approach leverages these explanations to improve the accuracy and robustness of text classification models.

## Repository Structure

- **`Dialectical_Prompting/`**: Contains the codebase, datasets, and configurations for the Dialectical Prompting approach. This directory includes scripts for running experiments, evaluating model performance, and generating explanations using LLMs.
  
- **`Explanation_Guided_Training/`**: Contains the codebase and configurations for the Explanation Guided Training approach. This directory includes scripts to process data, train models, and evaluate the effectiveness of explanation-guided training.

- **`README.md`**: The file you are currently reading, providing an overview of the repository and its contents.

## Execution Order

### Step 1: Run Dialectical Prompting

1. Navigate to the `Dialectical_Prompting` directory:
   ```bash
   cd Dialectical_Prompting
   ```

2. Follow the instructions in the `README.md` within the `Dialectical_Prompting` directory to generate data files with dialectical explanations. These explanations will be used in the next step.

3. The generated data files should be saved and then moved to the `datasets/` directory of the `Explanation_Guided_Training` project.

### Step 2: Run Explanation Guided Training

1. Navigate to the `Explanation_Guided_Training` directory:
   ```bash
   cd ../Explanation_Guided_Training
   ```

2. Place the data files with dialectical explanations generated from the previous step into the `datasets/` directory.

3. Follow the instructions in the `README.md` within the `Explanation_Guided_Training` directory to run the experiments using these datasets.

### Step 3: Evaluating the Results

1. If you want to evaluate the results from `Explanation_Guided_Training`, take the output files generated in the `output/` directory of `Explanation_Guided_Training` and move them to the `EQ-Evaluating-Input/` directory in the `Dialectical_Prompting` project.

2. Navigate back to the `Dialectical_Prompting` directory:
   ```bash
   cd ../Dialectical_Prompting
   ```

3. Run the `Evaluating_explanation_quality.py` script to assess the quality of the explanations:
   ```bash
   python Evaluating_explanation_quality.py
   ```

## Getting Started

### 1. Clone the Repository

To get started with the project, clone the repository to your local machine:

```bash
git clone <repository-url>
cd DET
```

### 2. Set Up the Environment

For each subdirectory, follow the environment setup instructions provided in their respective README files. Typically, this will involve setting up a Python virtual environment and installing dependencies.



## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
