import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from typing import List, Tuple

DATA_PATH = "./data/week4_dataset1.csv"
GRAPHS_PATH = "./graphs"


# Function to load the dataset
def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(filepath, header=None)
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    return X, y


# Function to perform cross-validation and find the best polynomial degree and C
def perform_grid_search(X: np.ndarray, y: np.ndarray, degrees: List[int], C_values: np.ndarray,
                        n_splits: int = 5) -> GridSearchCV:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipe = Pipeline([
        ('poly', PolynomialFeatures()),  # Add polynomial features
        ('logistic', LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000))
        # L2 regularized logistic regression
    ])

    param_grid = {
        'poly__degree': degrees,
        'logistic__C': C_values
    }

    grid_search = GridSearchCV(pipe, param_grid, cv=kf, scoring='accuracy', refit=True)
    grid_search.fit(X, y)

    return grid_search  # Return the entire GridSearchCV object


# Function to plot cross-validation results with error bars
def plot_cv_results(grid_search: GridSearchCV, save_path: str):
    results = grid_search.cv_results_  # Access cv_results_ from the GridSearchCV object
    mean_scores = results['mean_test_score']
    std_scores = results['std_test_score']

    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(mean_scores)), mean_scores, yerr=std_scores, fmt='o')
    plt.xticks(range(len(mean_scores)), labels=[f"Degree: {d}, C: {c:.4f}" for d, c in
                                                zip(results['param_poly__degree'], results['param_logistic__C'])],
               rotation=90)
    plt.ylabel("Mean CV Accuracy")
    plt.title("Cross-validation results (accuracy with error bars)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# Function to plot decision boundary
def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: Pipeline, degree: int, save_path: str):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # Train the model
    model.fit(X_poly, y)

    # Create mesh grid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(poly.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.title(f"Decision Boundary for Degree {degree}")
    plt.savefig(save_path)
    plt.show()


# Main function to run the pipeline
def run_pipeline(data_path: str, graphs_path: str, degrees: List[int], C_values: np.ndarray, n_splits: int = 5):
    # Load dataset
    X, y = load_dataset(data_path)

    # Perform grid search to find the best model and parameters
    grid_search = perform_grid_search(X, y, degrees, C_values, n_splits)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best parameters:", best_params)
    print("Best cross-validation score (accuracy):", best_score)

    # Plot cross-validation results
    cv_plot_path = f"{graphs_path}/cv_accuracy_plot.png"
    plot_cv_results(grid_search, cv_plot_path)

    # Plot decision boundary with the best polynomial degree
    best_degree = best_params['poly__degree']
    decision_boundary_path = f"{graphs_path}/decision_boundary_degree_{best_degree}.png"
    plot_decision_boundary(X, y, best_model, best_degree, decision_boundary_path)


# Example usage
if __name__ == "__main__":
    # Hyperparameter ranges
    degrees = [1, 2, 3, 4, 5]  # Polynomial degrees to test
    C_values = np.logspace(-4, 4, 10)  # Range of C values

    # Run the pipeline
    run_pipeline(DATA_PATH, GRAPHS_PATH, degrees, C_values)
