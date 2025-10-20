🏦 Consumer Complaint Text Classification

💡 Project Overview

This project is a text classification pipeline for consumer complaints submitted to financial institutions. It automatically classifies complaint texts into their respective financial categories, helping organizations improve response times and streamline operations.

Key Categories:

Label	Category
0	Credit reporting, credit repair, or other
1	Debt collection
2	Consumer Loan
3	Mortgage

Goal: Compare multiple machine learning models and find the best performing one for text classification.

🛠️ Technologies Used

Python 3.12

Pandas, NumPy – Data manipulation

NLTK – Text preprocessing (tokenization, stopword removal, lemmatization)

Scikit-learn – Feature extraction, model training, evaluation

Matplotlib & Seaborn – Data visualization

📂 Dataset

Source: Consumer Complaint Database

Columns Used:

Text – Consumer complaint text

Category – Complaint type (target variable)

🧹 Data Preprocessing

Text Cleaning: Remove non-alphabetic characters, lowercase conversion.

Tokenization: Split text into words.

Stopword Removal: Remove common English stopwords.

Lemmatization: Convert words to their base form.

Feature Engineering: TF-IDF vectorization (optionally combined with Bag-of-Words for Logistic Regression).

⚙️ Train-Test Split

Stratified 80/20 split to maintain category distribution.

Ensures no data leakage between training and testing datasets.

🤖 Models Implemented

Linear Support Vector Machine (SVM) – Best performer ✅

Naive Bayes (MultinomialNB)

Random Forest Classifier

Logistic Regression (with combined TF-IDF + BOW features)

📊 Results
Model	Accuracy
Linear SVM	0.9904
Naive Bayes	0.9816
Random Forest	0.9626
Logistic Regression	0.9626

Observation: Linear SVM achieved the highest accuracy while being computationally efficient.

📈 Visualizations

Confusion Matrices: Check per-class performance.

ROC Curves: One-vs-Rest evaluation for multi-class classification.

Accuracy Comparison Bar Chart: Visual comparison of all models.

🏃 How to Run

Clone repository:

git clone https://github.com/yourusername/consumer-complaint-classification.git
cd consumer-complaint-classification


Install dependencies:

pip install pandas numpy scikit-learn nltk matplotlib seaborn


Load dataset:

import pandas as pd
df = pd.read_csv('Consumer_Complaints.csv')


Run preprocessing pipeline:

# Clean, tokenize, remove stopwords, lemmatize
df['Clean_Text'] = df['Text'].apply(clean_function)
df['Tokenize_Text'] = df['Clean_Text'].apply(tokenize_function)
df['Nostopword_Text'] = df['Tokenize_Text'].apply(remove_stopwords)
df['Lemmatized_Text'] = df['Nostopword_Text'].apply(lemmatize_function)


Train-Test Split:

from sklearn.model_selection import train_test_split
X = [' '.join(tokens) for tokens in df['Lemmatized_Text']]
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


Train and evaluate models (Example: Linear SVM):

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.8, min_df=10, max_features=2000)),
    ('svm', LinearSVC(max_iter=5000))
])
pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


Compare all models:

import matplotlib.pyplot as plt

models = ["Random Forest", "Linear SVM", "Naive Bayes", "Logistic Regression"]
accuracies = [accuracy_rf, accuracy_svm, accuracy_nb, accuracy_lr]

plt.figure(figsize=(10,6))
colors = ['purple', 'blue', 'green', 'red']
plt.bar(models, accuracies, color=colors)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()

⚡ Key Takeaways

Linear SVM: Fast training, high accuracy – best choice for large text datasets.

Random Forest: Works but slower for high-dimensional TF-IDF features.

Naive Bayes: Simple and fast, slightly lower accuracy.

Logistic Regression: Good performance, benefits from combining BOW + TF-IDF.

📖 References

Consumer Complaint Database - data.gov

Scikit-learn Documentation

NLTK Documentation

✅ Project Status: Complete
🏆 Best Model: Linear SVM (Accuracy 0.9904)
