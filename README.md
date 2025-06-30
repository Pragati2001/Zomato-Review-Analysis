# 🧠 Zomato Review Sentiment Analysis using NLP and Gaussian Naive Bayes

## 📌 Project Overview

This project is a machine learning-based sentiment analysis system that predicts customer sentiment or review ratings based on textual data collected from the **Zomato Restaurant Review Dataset**. Using **Natural Language Processing (NLP)** and a **Gaussian Naive Bayes classifier**, we convert raw user reviews into a structured format suitable for machine learning, train a model, and evaluate its performance.

---

## 🎯 Objectives

- Clean and preprocess unstructured textual review data
- Convert text into a numerical format (Bag of Words)
- Train a machine learning model to classify sentiment or rating
- Evaluate the model using confusion matrix and accuracy metrics

---

## 🧰 Tools & Libraries Used

| Tool/Library       | Purpose                             |
|--------------------|-------------------------------------|
| **Python**         | Core programming language           |
| **Pandas**         | Data manipulation                   |
| **NumPy**          | Numerical computations              |
| **NLTK**           | Natural language processing         |
| **Scikit-learn**   | ML algorithms, vectorization, metrics |
| **Regex** (`re`)   | Text cleaning and pattern matching  |

---

## 📁 Dataset

- File: `Dataset_master - Zomato Review Kaggle.csv`
- Description: Contains restaurant reviews with corresponding user ratings.
- Format: CSV
- Key Column Used: `'Review'` for text, last column as the target variable (e.g., sentiment or rating)

---

## 🔄 Data Preprocessing Steps

```python
# Remove non-alphabet characters using regex
# Convert text to lowercase
# Remove stopwords (except "not" to preserve sentiment)
# Apply stemming to normalize words
```

---

## 🧠 Feature Extraction (Bag of Words)

We use `CountVectorizer` from scikit-learn to:

- Convert the corpus into a sparse matrix of word counts
- Limit the vocabulary size to the top 1600 most frequent words
- Transform reviews into numerical vectors (`X`) for modeling

```python
cv = CountVectorizer(max_features=1600)
X = cv.fit_transform(corpus).toarray()
y = zomato_csv.iloc[:, -1].values  # Extract the target label
```
---

## 🧪 Model Training
Algorithm: Gaussian Naive Bayes (GNB)
We chose GNB for the following reasons:

- Efficient for high-dimensional feature spaces like text

- Works well with Bag of Words features (assumes Gaussian distribution of features)

- Simple and interpretable

```python
Copy
Edit
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

---

## 📊 Evaluation
We evaluate the model on unseen test data using:

- Confusion Matrix – to observe classification performance per class

- Accuracy Score – percentage of correct predictions

```python
Copy
Edit
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

✅ Sample Output:
lua
Copy
Edit
Confusion Matrix:
[[73 15]
 [12 100]]

Accuracy: 86.5%
```
Results may vary depending on your dataset and random state.
![image](https://github.com/user-attachments/assets/bd29bbbf-2182-4e52-a916-d860af52711f)


---

🚀 How to Run
- Clone the repository

- Place the dataset CSV in the root folder

- Run the Python file or Jupyter notebook step by step

- Ensure required libraries are installed (pip install nltk pandas scikit-learn)

- pip install -r requirements.txt

- Make sure to download NLTK stopwords:

```python
Copy
Edit
import nltk
nltk.download('stopwords')
```

---

## 💬 Conclusion
This project demonstrates how traditional machine learning techniques like Naive Bayes can be effectively used for text classification. It also highlights the importance of proper NLP preprocessing in determining model performance.

Even simple models, when trained on clean, well-prepared data, can yield powerful and interpretable results in real-world applications.

---

## Example Use Cases
Predicting user satisfaction from restaurant reviews

Monitoring feedback in food delivery platforms

Sentiment filtering for customer service pipelines

