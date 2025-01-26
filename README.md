# Amazon Office Products Review - ML Model Comparison Project 
# Surya Pugazhenthi
# October - December 2023

## Overview

This project aims to compare the performance of two machine learning models for natural language analysis using product reviews from the "Office Products" category on Amazon. The models included in this study are:

1. **Simple Linear Regression Model**
2. **Advanced Random Forest Classification Model**

The project leverages Jianmo Ni's 2018 "small" subset of the Amazon Office Products dataset, which includes product reviews used as training data.

---

## Objectives

1. To preprocess and analyze text data for language-based insights.
2. To build and train two distinct machine learning models:
   - A simple linear regression model.
   - A more complex random forest classification model.
3. To evaluate and compare the models using performance metrics such as accuracy, precision, recall, and F1-score.
4. To determine the suitability of each model for this specific task.

---

## Dataset

### Source

The dataset was obtained from Jianmo Ni's repository: [Amazon Office Products Dataset](https://nijianmo.github.io/amazon/)

### Description

- **Category:** Office Products
- **Type:** "Small" subset
- **Content:**
  - Product reviews
  - Metadata (e.g., ratings, product descriptions, review text, etc.)

---

## Project Workflow

### 1. Data Collection and Exploration

- Download and load the "small" subset file of Amazon Office Products reviews.
- Explore the dataset to understand its structure, content, and distribution of key variables such as review scores and word counts.

### 2. Data Preprocessing

- Handle missing values and inconsistencies in the dataset.
- Perform text cleaning:
  - Removing special characters and stopwords.
  - Lowercasing and stemming/lemmatization.
- Tokenization of review text for NLP-based analysis.
- Feature engineering:
  - Extracting sentiment polarity scores.
  - Calculating word frequency and other linguistic features.

### 3. Model Development

#### Model 1: Simple Linear Regression

- Use numerical features derived from text analysis to train the linear regression model.
- Evaluate performance metrics (e.g., Mean Squared Error, R-squared).

#### Model 2: Random Forest Classification

- Utilize both text-based features and sentiment analysis.
- Train the model using a classification approach.
- Evaluate performance metrics (e.g., Accuracy, Precision, Recall, F1-score).

### 4. Model Evaluation

- Compare the performance of both models using appropriate metrics.
- Discuss strengths, weaknesses, and potential use cases for each model.

### 5. Results and Analysis

- Summarize the findings and provide visualizations (e.g., confusion matrices, performance graphs).
- Provide insights into the effectiveness of each model for language analysis tasks.
- Results and author's thoughts included in report

---

## Key Files

1. Office_Products_Reviews_Data.json: Contains the original dataset file and preprocessed data.
2. LinearRegression.py: Python file for linear regression model.
3. RandomForest.py: Python file for random forest model.
4. README.md: This document.

---

## How to Run

### Prerequisites

1. Install Python 3.8 or later.
2. Install the required packages using the following command:
   ```bash
   pip install -r requirements.txt
   ```

### Steps

1. Download the dataset from the provided link and save it to the respective ML model directory.
2. Run the python file in an IDE with the dataset file.
3. Proceed with console application running.
4. View the results and evaluate the model based on its provided metrics in the console.
5. Repeat steps with other ML model.
6. Evaluate both models and recognize pros/cons for each.
7. Go on with your day or read my report, I don't mind either decision.


---

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - Pandas
  - Scikit-learn
  - NLTK
---

## Results and Discussion

The performance metrics for both models are summarized in my report.

## Contributors

- [Surya Pugazhenthi] (Report author and project developer)
- [Nick Crawford] (Discrete Math 160 Professor at Los Medanos College who supervised my project and its report) 

---

## License

This project is NOT to be replicated for classwork. Please email me for further inquires: suryapugaz1629@gmail.com



Thank you for reading this document :)

