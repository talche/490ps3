# 490ps3: Heart Disease Prediction with Feed-forward Neural Network

## Overview

This repository contains an implementation of a simple feed-forward neural network built from scratch using Python and NumPy. The neural network features two hidden layers, allowing it to effectively model complex relationships within the data. It is designed to be modular, enabling users to adapt and extend it for various machine learning tasks.

The model is applied to a heart disease classification dataset, which determines whether or not a person has heart disease based on several health-related factors. This showcases the network's application in real-world scenarios and its capability for binary classification tasks.


## Dataset

The dataset used in this project is the [Heart Disease Dataset](https://huggingface.co/datasets/muhrafli/heart-diseases) available on Hugging Face. Make sure to download it and save it as `heart.csv` in the same directory as the script.

## Files

- **heart.csv**: The dataset used for training and testing the neural network. This dataset contains various health metrics used for predicting heart disease.
- **neural_network.py**: The Python file containing the `NeuralNetwork` class, which includes methods for initialization, forward propagation, backpropagation, training, and prediction.
- **build_dataset.py**: The Python file containing the `BuildDataset` function, which preprocesses and splits the given dataset into testing and training data.
- **ps3main.ipynb**: A Jupyter Notebook demonstrating the use of the `NeuralNetwork` class with the `heart.csv` dataset. It includes the full training process, evaluation, and example predictions.
- **README.md**: This file, explaining the project and usage details.

## Usage

### Prerequisites

Ensure you have Python 3.x installed along with the following libraries:

```bash
pip install numpy pandas matplotlib jupyter
```

### Running the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **Open and Run the Jupyter Notebook**:
   ```bash
   jupyter notebook NeuralNetwork.ipynb
   ```

## Dataset

The dataset used in this project, `heart.csv`, contains multiple samples, each representing a unique patient profile with 12 health-related features. The classification target indicates whether the patient has heart disease (1) or does not (0).

### Features

The dataset comprises the following factors:

- **Age**: Age of the patient.
- **Sex**: Gender of the patient (M/F).
- **ChestPainType**: Type of chest pain experienced.
- **RestingBP**: Resting blood pressure (mm Hg).
- **Cholesterol**: Serum cholesterol (mg/dl).
- **FastingBS**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
- **RestingECG**: Resting electrocardiographic results (Normal/ST/Hypertrophy).
- **MaxHR**: Maximum heart rate achieved.
- **ExerciseAngina**: Exercise induced angina (Y/N).
- **Oldpeak**: ST depression induced by exercise relative to rest.
- **ST_Slope**: Slope of the peak exercise ST segment.

### Target Variable

- **HeartDisease**: The output is binary, where 0 indicates a patient who does not have heart disease, and 1 indicates a patient who does.


This is for the 490 ps3 assignment and for learning purposes.
