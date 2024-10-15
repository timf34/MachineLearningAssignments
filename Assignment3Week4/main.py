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


df = read_data(DATA_PATH)
X1, X2, y = parse_data(df)

# Plot original dataset
plot_dataset(X1, X2, y, x_label='Feature 1', y_label='Feature 2', title='Original Dataset')

# Augment features
X_augmented = augment_features(X1, X2, degree=3)

# Plot first two dimensions of augmented dataset
plot_dataset(X_augmented[:, 0], X_augmented[:, 1], y, 
             x_label='Augmented Feature 1', y_label='Augmented Feature 2', 
             title='Augmented Dataset (First 2 Dimensions)')

print(f"Original feature shape: {X1.shape[0]}x2")
print(f"Augmented feature shape: {X_augmented.shape}")
