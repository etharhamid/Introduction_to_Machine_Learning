# Introduction to Machine Learning: From Scratch

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Library](https://img.shields.io/badge/library-NumPy-orange)
![Status](https://img.shields.io/badge/status-educational-yellow)

## 📖 Overview

This repository contains a collection of fundamental Machine Learning algorithms implemented **from scratch** using Python and NumPy. The goal of this project is to deconstruct complex models into their mathematical components to understand *how* they work under the hood, rather than simply relying on high-level frameworks.

From simple linear regressors to multi-layer neural networks, these notebooks walk through the mathematics of optimization, cost functions, and backpropagation step-by-step.

## 🚀 Key Concepts Covered

- **Optimization**: Manual implementation of Gradient Descent loops.
- **Loss Functions**: Deep dives into Mean Squared Error (MSE), Cross-Entropy, and Gini Index.
- **Model Architecture**: Building neurons, layers, and decision trees variable by variable.
- **Data Engineering**: Manual feature scaling, normalization, and matrix augmentation.
- **Regularization**: Exploring techniques like Dropout to prevent overfitting.

## 📂 Repository Structure

| Notebook | Description | Key Concepts |
| :--- | :--- | :--- |
| `Linear_regression.ipynb` | **Linear Regression**: The "Hello World" of ML. Implements a regressor to predict continuous values using iterative optimization. | • Hypothesis Function<br>• Cost Function (MSE)<br>• Gradient Descent<br>• Residual Analysis |
| `Linear_Classifier.ipynb` | **Linear Classification**: A binary classifier implemented by hand. Focuses heavily on data preparation and feature engineering. | • Feature Scaling & Normalization<br>• Matrix Augmentation<br>• Binary Labels<br>• Training/Test Split |
| `Decision_Stump.ipynb` | **Decision Stump**: A one-level Decision Tree. This notebook breaks down how trees make splitting decisions. | • Gini Index Calculation<br>• Cost Function for Splits<br>• Threshold Selection<br>• Visualization |
| `Neural_Network.ipynb` | **Neural Network**: A fully connected network built with NumPy matrices. Implements the full forward and backward passes. | • Weighted Sums<br>• Activation Functions (Sigmoid/ReLU)<br>• Backpropagation (Derivatives)<br>• Cross-Entropy Loss |
| `NN_Dropout_Regularization.ipynb` | **Regularization**: Explores advanced neural network concepts using framework comparisons to demonstrate the impact of Dropout. | • Overfitting vs. Underfitting<br>• Dropout Layers<br>• Model Architecture Design |

## 🛠️ Tech Stack

- **Core Logic**: `NumPy` (Matrix multiplication, math operations)
- **Data Handling**: `Pandas` (DataFrames, loading datasets)
- **Visualization**: `Matplotlib`, `Seaborn`
- **Frameworks**: `TensorFlow/Keras` (Used specifically in the Regularization notebook for demonstration)

## ⚡ Getting Started

### Prerequisites

You will need Python installed along with the standard data science suite. You can install the dependencies via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
