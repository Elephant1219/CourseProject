from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'results/model.joblib')
    return model, X_test, y_test
