import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from typing import List, Tuple


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

    return grid_search


# Function to plot cross-validation results with error bars
def plot_cv_results(grid_search: GridSearchCV, save_path: str):
    results = grid_search.cv_results_
    mean_scores = results['mean_test_score']
    std_scores = results['std_test_score']

    plt.figure(figsize=(12, 6))
    for degree in set(results['param_poly__degree']):
        degree_mask = results['param_poly__degree'] == degree
        C_values = results['param_logistic__C'][degree_mask]
        scores = mean_scores[degree_mask]
        errors = std_scores[degree_mask]
        plt.errorbar(C_values, scores, yerr=errors, fmt='o-', label=f'Degree {degree}')

    plt.xscale('log')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('Mean CV Accuracy')
    plt.title('Cross-validation results (accuracy with error bars)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Function to plot decision boundary
def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: Pipeline, save_path: str):
    # Create mesh grid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(
        f"Decision Boundary (Degree: {model.named_steps['poly'].degree}, C: {model.named_steps['logistic'].C:.4f})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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

    # Plot decision boundary with the best model
    decision_boundary_path = f"{graphs_path}/decision_boundary.png"
    plot_decision_boundary(X, y, best_model, decision_boundary_path)


# Example usage
if __name__ == "__main__":
    DATA_PATH = "./data/week4_dataset1.csv"
    GRAPHS_PATH = "./graphs/"

    # Hyperparameter ranges
    degrees = [1, 2, 3, 4, 5]  # Polynomial degrees to test
    C_values = np.logspace(-4, 4, 10)  # Range of C values

    # Run the pipeline
    run_pipeline(DATA_PATH, GRAPHS_PATH, degrees, C_values)