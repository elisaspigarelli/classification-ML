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
├── Page_Blocks_Classification.py    # Python script to train and evaluate models
├── /project-document
│   └── documentation.pdf           # Project report with details and choices made
├── README.md                       # This file

```
## Conclusion

The project demonstrates how machine learning can be applied to classify page blocks in a dataset. By using different models and evaluating their performance, we were able to identify the best approach for this classification task.

A full explanation of the project, including the methodology, model choices, and results, can be found in the project documentation.
