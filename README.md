# Image Classification Using Naive Bayes, Decision Trees, MLP, and CNN on CIFAR-10 Dataset.

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
* Saved models: for all models and any variants
* Report: Detailed explanations of methods and evaluation.
## How to Use

  1. Clone the Repository
     ```bash
          git clone https://github.com/Anfloe/Comp472-Team-solo.git
          cd Comp472-Team-solo
     ```
           

  3. Install Required Libraries\
      Ensure you have Python installed (version 3.8 or above).\
      Install the required libraries using the provided requirements.txt file:
  ```bash
        pip install -r requirements.txt
  ```
## General Instructions
To run and load trained models in the provided notebook please place the trained model files in the same working directoy as the jupyter notebook.
The data loads automatically when running the notebook from the PyTorch library online, so an internet connection is needed.
For MLP and CNN GPU is preferred.

To run the trained models in the current Notebook: 

* For each model:\
		* Run the data loading and preprocessing cells\
		* Navigate to the model section in the notebook\
		* Run the cells that define and initiate the models\
		* Navigate to the model evaluation section and run the cells that load the pretrained model and perform predictions and calculate metrics

**Warning** Loading and preprocessing data is different fot CNNs and for the very last model: vgg_var_model

## The saved models:

**Naive Bayes:**
The custom model and the variant for comparison using sklearn:\
Filenames:\
		NaiveBayes_custom_model.pkl\
		sk_learn_Gauss_model.pkl

**Decision trees**
The custom model with best depth and the variant for comparison sing sklearn:\
Filenames:\
best_custom_tree.pkl  \
best_sk_learn_tree.pkl

**MLP**
The model with best depth and hidden size\
Filename:\
mlp_model.pth

**CNN**
Saved four variants: 

Best: with kernel variations:\
Filename: vgg_var_model.pth

Base VGG11:\
Filename: vgg_model.pth
	
Shallow CNN:\
Filename: vgg_small_model.pth

Deeper CNN:\
Filename: vgg_big_model.pth
	
