import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path='data/movies.csv'):
    df = pd.read_csv(path)
    df = df.dropna(subset=['Plot', 'Genre'])
    df = df.drop_duplicates(subset='Plot')
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Plot'])
    le = LabelEncoder()
    y = le.fit_transform(df['Genre'])
    return X, y, tfidf, le
