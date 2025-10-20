consumer Complaint Text Classification
Project Overview

This project focuses on text classification of consumer complaints submitted to financial institutions. The goal is to automatically categorize complaint texts into their corresponding financial category, enabling faster processing and better customer service.

Categories:

Label	Category
0	Credit reporting, credit repair, or other
1	Debt collection
2	Consumer Loan
3	Mortgage

The project demonstrates data preprocessing, feature engineering, multiple classification models, and model comparison.

Dataset

Source: Consumer Complaint Database - data.gov

Main columns used:

Text – consumer complaint content

Category – complaint type (target variable)

Steps Performed

Data Loading & Exploration

Extracted and loaded the CSV file from a ZIP.

Checked for missing values.

Computed basic statistics: number of characters, words, and sentences per complaint.

Text Cleaning & Preprocessing

Lowercasing and removing non-alphabetic characters.

Tokenization.

Stopword removal.

Lemmatization.

Feature Engineering

Created a corpus from lemmatized tokens.

Vectorized text using TF-IDF and optionally combined with Bag-of-Words (BOW) for Logistic Regression.

Train-Test Split

Stratified 80/20 split to maintain category distribution.

Ensured no data leakage between train and test sets.

Model Building
Trained the following models:

Naive Bayes (MultinomialNB)

Linear SVM (LinearSVC) – best performing model

Random Forest (RandomForestClassifier)

Logistic Regression (with FeatureUnion of BOW + TF-IDF)

Model Evaluation

Accuracy score

Classification report (precision, recall, F1-score)

Confusion matrix

ROC curves (One-vs-Rest)

Comparison of all models using bar charts

Results
Model	Accuracy
Linear SVM	0.9904 ✅
Naive Bayes	0.9816
Random Forest	0.9626
Logistic Regression	0.9626

Linear SVM performed best, combining high accuracy with efficient training time on the dataset.

How to Run

Install dependencies:

pip install pandas numpy scikit-learn nltk matplotlib seaborn


Load the dataset:

import pandas as pd
df = pd.read_csv('Consumer_Complaints.csv')


Preprocess text:

# Clean, tokenize, remove stopwords, lemmatize
df['Clean_Text'] = df['Text'].apply(clean_function)
df['Tokenize_Text'] = df['Clean_Text'].apply(tokenize_function)
df['Nostopword_Text'] = df['Tokenize_Text'].apply(remove_stopwords)
df['Lemmatized_Text'] = df['Nostopword_Text'].apply(lemmatize_function)


Train-test split:

from sklearn.model_selection import train_test_split
X = [' '.join(tokens) for tokens in df['Lemmatized_Text']]
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


Train models:

# Example: Linear SVM
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.8, min_df=10, max_features=2000)),
    ('svm', LinearSVC(max_iter=5000))
])
pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)


Evaluate models:

from sklearn.metrics import accuracy_score, classification_report
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


Compare all models:

import matplotlib.pyplot as plt
models = ["Random Forest", "Linear SVM", "Naive Bayes", "Logistic Regression"]
accuracies = [accuracy_rf, accuracy_svm, accuracy_nb, accuracy_lr]

plt.figure(figsize=(10,6))
plt.bar(models, accuracies, color=['purple','blue','green','red'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()

Notes

Linear SVM is recommended for large text datasets because it offers fast training and high accuracy.

Random Forest can be slow with high-dimensional TF-IDF features.

Naive Bayes is fast and simple but slightly less accurate than SVM or Logistic Regression.
