import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from typing import List, Tuple

DATA_PATH = "./data/week4_dataset1.csv"
GRAPHS_PATH = "./graphs/"


def read_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path, header=None)


def parse_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    X1 = df.iloc[:, 0]
    X2 = df.iloc[:, 1]
    y = df.iloc[:, 2]
    return X1, X2, y


def plot_dataset(
        X1: np.ndarray,
        X2: np.ndarray,
        y: np.ndarray,
        x_label: str = 'X1',
        y_label: str = 'X2',
        title: str = 'Dataset Plot'
) -> None:
    """
    Plot a dataset given two feature columns and a target column.

    Args:
    X1 (np.ndarray): First feature column
    X2 (np.ndarray): Second feature column
    y (np.ndarray): Target column
    x_label (str): Label for x-axis
    y_label (str): Label for y-axis
    title (str): Title of the plot
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X1, X2, c=y, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{GRAPHS_PATH}{title.lower().replace(' ', '_')}.png")
    plt.close()


def augment_features(X1: np.ndarray, X2: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Augment the original two features with polynomial features.
    """
    X = np.column_stack((X1, X2))
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly


def train_logistic_regression_cv(X: np.ndarray, y: np.ndarray) -> Tuple[Pipeline, dict]:
    """
    Train a Logistic Regression classifier with L2 regularization using cross-validation
    to select the best polynomial degree and regularization strength.

    Args:
    X (np.ndarray): Input features
    y (np.ndarray): Target variable

    Returns:
    Tuple[Pipeline, dict]: Best estimator and best parameters
    """
    # Define the pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(include_bias=False)),
        ('clf', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000))
    ])

    # Define the parameter grid
    param_grid = {
        'poly__degree': [1, 2, 3, 4],  # Polynomial degrees to try
        'clf__C': np.logspace(-4, 4, 20)  # Regularization strengths to try
    }

    # Set up k-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    # Get the best estimator and parameters
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best parameters:", best_params)
    print("Best cross-validation score:", grid_search.best_score_)

    return best_estimator, best_params


df = read_data(DATA_PATH)
X1, X2, y = parse_data(df)

# # Plot original dataset
# plot_dataset(X1, X2, y, x_label='Feature 1', y_label='Feature 2', title='Original Dataset')

# Augment features
# X_augmented = augment_features(X1, X2, degree=3)
#
# # Plot first two dimensions of augmented dataset
# plot_dataset(X_augmented[:, 0], X_augmented[:, 1], y,
#              x_label='Augmented Feature 1', y_label='Augmented Feature 2',
#              title='Augmented Dataset (First 2 Dimensions)')
#
# print(f"Original feature shape: {X1.shape[0]}x2")
# print(f"Augmented feature shape: {X_augmented.shape}")

X = np.column_stack((X1, X2))  # Combine original features
best_model, best_params = train_logistic_regression_cv(X, y)
