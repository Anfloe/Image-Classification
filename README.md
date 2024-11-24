# Image Classification Using Naive Bayes, Decision Trees, MLP, and CNN on CIFAR-10 Dataset.
Author: Andreea Florean
Course: Comp 472, Fall 2024

## Introduction
This project explores image classification on the CIFAR-10 dataset using both statistical and deep learning models. The implemented models include:

* Naïve Bayes (NB) and Decision Tree (DT): Built from scratch with NumPy and compared with Scikit-learn implementations.
* Multilayer Perceptron (MLP): Developed in PyTorch with feature extraction using ResNet-18 and PCA, followed by architectural tuning.
* Convolutional Neural Networks (CNNs): Based on the VGG-11 architecture with variations in depth and kernel sizes.
## Features
* Feature Extraction: Used ResNet-18 to generate 512-dimensional feature vectors, reduced to 50 dimensions using PCA.
* Hyperparameter Tuning: Grid search for batch sizes and learning rates, with early stopping for training optimization.
* Evaluation Metrics: Accuracy, precision, recall, F1-score, confusion matrices, and classification reports.
## Results
* Best Performing Model: MLP achieved the highest test accuracy of 81.9%, leveraging ResNet-18 features.
* CNN Variants: A modified CNN with larger kernels achieved 69.5% test accuracy but faced overfitting challenges.
## Repository Contents
* Jupyter Notebook: Implementation of models, feature extraction, and training scripts.
* Saved models: Confusion matrices, classification reports, and performance metrics.
* Report: Detailed explanations of methods, hyperparameter tuning, and evaluation.
## How to Use

  1. Clone the Repository
     ```bash
          git clone https://github.com/Anfloe/Comp472-Team-solo.git
          cd Comp472-Team-solo
     ```
           

  3. Install Required Libraries
      Ensure you have Python installed (version 3.8 or above). Install the required libraries using the provided requirements.txt file:
  ```bash
        pip install -r requirements.txt
  ```
