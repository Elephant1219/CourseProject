# Movie Genre Prediction from Plot Summaries

## 📌 Problem
This project builds a classifier that predicts a movie’s genre based on its plot summary. It helps organize content and build smarter recommender systems.

## 🧠 Model
We use TF-IDF to transform plot text into features and train a Decision Tree classifier to predict genres.

## 📊 Results
- Accuracy: ~60%
- Precision/Recall: varies by genre
- Common confusions: Comedy ↔ Drama

## 📈 Improvements
- Try Naive Bayes or SVM
- Expand to multi-label classification
- Use word embeddings instead of TF-IDF
