"""Bank Note Authentication - Function-Based Script."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data(filepath: str) -> pd.DataFrame:
	"""Load the dataset from a CSV file.

	Args:
		filepath (str): _description_

	Returns:
		pd.DataFrame: _description_

	"""
	return pd.read_csv(filepath)


def explore_data(df: pd.DataFrame) -> None:
	"""Print basic information and head of the dataframe.

	Args:
		df (pd.DataFrame): _description_

	"""
	print('First 5 rows of the dataset:')
	print(df.head())
	print('Dataset Info:')
	print(df.info())
	print('Statistical Summary:')
	print(df.describe())


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Separate features and target.

	Args:
		df (pd.DataFrame): _description_

	Returns:
		pd.DataFrame: _description_

	"""
	x = df.iloc[:, :-1]
	y = df.iloc[:, -1]
	return x, y


def train_model(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
	"""Train a Random Forest model and return it.

	Args:
		x (pd.DataFrame): _description_
		y (pd.DataFrame): _description_

	Returns:
		pd.DataFrame: _description_

	"""
	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=0.3,
		random_state=0,
	)
	classifier = RandomForestClassifier()
	classifier.fit(x_train, y_train)
	y_pred = classifier.predict(x_test)
	acc = accuracy_score(y_test, y_pred)
	print(f'Model Accuracy: {acc:.4f}')
	return classifier


def main() -> None:
	filepath = 'BankNote_Authentication.csv'
	df = load_data(filepath)
	explore_data(df)
	x, y = preprocess_data(df)
	train_model(x, y)


if __name__ == '__main__':
	main()
