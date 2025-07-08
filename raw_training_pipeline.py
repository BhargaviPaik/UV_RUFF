"""
Bank Note Authentication - Function-Based Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(filepath: str):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def explore_data(df):
    """Print basic information and head of the dataframe."""
    print("First 5 rows of the dataset:")
    print(df.head())
    print("Dataset Info:")
    print(df.info())
    print("Statistical Summary:")
    print(df.describe())

def preprocess_data(df):
    """Separate features and target."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def train_model(X, y):
    """Train a Random Forest model and return it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    return classifier

def main():
    filepath = 'BankNote_Authentication.csv'
    df = load_data(filepath)
    explore_data(df)
    X, y = preprocess_data(df)
    train_model(X, y)

if __name__ == "__main__":
    main()
