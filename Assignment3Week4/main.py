import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Tuple, Dict, Any

DATA_PATH = "./data/week4_dataset1.csv"
GRAPHS_PATH = "./graphs/"

def read_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Read the CSV data file."""
    return pd.read_csv(data_path, header=None)

def parse_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and target from the DataFrame."""
    X = df.iloc[:, :2].values
    y = df.iloc[:, 2].values
    return X, y

def plot_dataset(
    X: np.ndarray,
    y: np.ndarray,
    x_label: str = 'X1',
    y_label: str = 'X2',
    title: str = 'Dataset Plot'
) -> None:
    """Plot a dataset given two feature columns and a target column."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f"{GRAPHS_PATH}{title.lower().replace(' ', '_')}.png")
    plt.close()


def create_pipeline() -> Pipeline:
    """Create a pipeline with polynomial features, scaling, and logistic regression."""
    return Pipeline([
        ('poly', PolynomialFeatures(include_bias=False)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000))
    ])


def train_logistic_regression_cv(X: np.ndarray, y: np.ndarray) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train a Logistic Regression classifier with L2 regularization using cross-validation
    to select the best polynomial degree and regularization strength.
    """
    pipeline = create_pipeline()

    param_grid = {
        'poly__degree': [1, 2, 3, 4],
        'clf__C': np.logspace(-4, 4, 20)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: Pipeline, title: str) -> None:
    """Plot the decision boundary of the model."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.savefig(f"{GRAPHS_PATH}{title.lower().replace(' ', '_')}.png")
    plt.close()


def ia():
    # Read and parse data
    df = read_data(DATA_PATH)
    X, y = parse_data(df)

    # Plot original dataset
    plot_dataset(X, y, x_label='Feature 1', y_label='Feature 2', title='Original Dataset')

    # Train model with cross-validation (using polynomial features)
    best_model, best_params = train_logistic_regression_cv(X, y)

    # Plot decision boundary
    plot_decision_boundary(X, y, best_model, 'Decision Boundary')

    # Evaluate model
    y_pred = best_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"Final model accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Compare original and augmented feature shapes
    best_degree = best_params['poly__degree']
    poly = PolynomialFeatures(degree=best_degree, include_bias=False)
    X_augmented = poly.fit_transform(X)
    print(f"Original feature shape: {X.shape}")
    print(f"Augmented feature shape: {X_augmented.shape}")
    print(f"Best polynomial degree: {best_degree}")


if __name__ == "__main__":
    ia()