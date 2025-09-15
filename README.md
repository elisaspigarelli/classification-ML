# Page Blocks Classification Project

## Overview

This project was completed as part of the Master's degree in Computer Engineering and Robotics - Data Science at Università degli Studi di Perugia. 
The goal was to develop a machine learning classification model to predict various categories in a Page Blocks Dataset. 
The dataset consists of multiple features related to different types of text blocks on scanned pages.

We built a Python script that applies different machine learning algorithms to classify these page blocks, exploring various techniques and evaluating their performance. 
Additionally, a comprehensive project document has been provided that explains the choices made during the development process, model selection, and evaluation.

## Project Description

The aim of this project was to classify page blocks from a dataset containing scanned page images. 
Each page block represents a region of text (or non-text) on a scanned page, and the task was to classify these blocks into different categories based on the provided features.

In this project, a Python script was developed that:

- Loads and preprocesses the dataset.
- Trains multiple machine learning models.
- Evaluates and compares the models' performance.
- Provides insights into model performance through various metrics (accuracy, precision, recall, etc.).

A detailed project document is also included, which explains the steps taken, from data preprocessing to model evaluation, with a particular focus on the choices made regarding the algorithms and techniques used.

## Data Description

The dataset used in this project is the Page Blocks Dataset. It contains several instances where each instance represents a block of text on a page. 
Each block has features like shape, color, and position, which can help determine the type of block (e.g., text block, image, table, etc.).

For more details about the dataset, refer to the project document provided.

## Project Structure

The project repository has the following structure:
```sh
/classification-ML
│
├── /page_block_data
│   └── page_blocks.data             # The dataset used for classification
├── page_blocks_classification.py    # Python script to train and evaluate models
├── /project-document
│   └── documentation.pdf           # Project report with details and choices made
├── README.md                       # This file

```
## Installation
To get started with this project, clone the repository and install the required dependencies.

Clone the repository:
```sh
git clone https://github.com/elisaspigarelli/classification-ML.git
cd classification-ML
```

### Install dependencies:
This project requires Python 3.x and several libraries. You can install the dependencies by running:
```sh
pip install -r requirements.txt
```

### Usage
To run the classification model, execute the Python script page_blocks_classification.py.

```sh
python page_blocks_classification.py
```

The script will load the dataset, preprocess the data, train multiple classification models (e.g., Softmax Regression, Support Vector Machine, Multi-layer Perceptron), and evaluate their performance.

Output:
The script will output the evaluation metrics for each model, such as accuracy, precision, recall, and confusion matrix, to help assess the performance.

## Model Selection

Different machine learning algorithms were tested to find the best-performing model for this classification task. Models tested include:
- k-Nearest Neighbours (k-NN): A simple, instance-based learning algorithm that classifies samples based on the majority label of their nearest neighbors.
- Softmax Regression: A generalization of logistic regression for multi-class classification.
- Support Vector Machine (SVM): A powerful classifier that works well in high-dimensional spaces.
- Multi-layer Perceptron (MLP): A feedforward neural network capable of capturing complex nonlinear relationships.
- Naive Bayes: A probabilistic classifier based on Bayes' theorem.

The choice of algorithms was based on their ability to handle the dataset's complexity and their performance on classification tasks.
For more details about the machine learning algorithms and their hyper parameters, refer to the project document provided.

## Evaluation
After training and evaluating multiple models, the following metrics were used to assess the models:

- Accuracy: The percentage of correctly classified instances, calcutated by balanced_accuracy_score function.
- Precision: The proportion of true positives among all positive predictions.
- Recall: The proportion of true positives among all actual positives.
- F1-Score: The harmonic mean of precision and recall, useful for imbalanced datasets.

The models were compared based on these metrics to select the best-performing model for the task.
 
  <img width="631" height="217" alt="Screenshot_20250915_090540" src="https://github.com/user-attachments/assets/29f44b5b-9425-4460-a13d-1e7e6ae6bd51" />

## Conclusion

The project demonstrates how machine learning can be applied to classify page blocks in a dataset. By using different models and evaluating their performance, we were able to identify the best approach for this classification task.

A full explanation of the project, including the methodology, model choices, and results, can be found in the project documentation.
