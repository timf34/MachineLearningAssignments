import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Tuple

DATA_PATH = "./data/week3.csv"
GRAPHS_PATH = "./graphs/"


def read_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path, header=None)


def parse_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Extract features (assuming all columns except the last are features)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def create_3d_scatter_plot(data_path: str, save_path: str, filename: str, x_label: str = 'Feature 1',
                           y_label: str = 'Feature 2', z_label: str = 'Target') -> None:
    data = read_data(data_path)
    X, y = parse_data(data)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with the first feature on the x-axis, second feature on the y-axis, and target on the z-axis
    ax.scatter(X[:, 0], X[:, 1], y)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Adjust the view angle to look directly along Feature 2 (y-axis)
    # ax.view_init(elev=0, azim=0)

    # Save the plot to the specified path
    save_path = os.path.join(save_path, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"3D scatter plot saved to {save_path}")


def train_and_analyze_model(model_type: str, data_path: str) -> None:
    """
    Train and analyze either Lasso or Ridge regression model
    """
    data = read_data(data_path)
    X, y = parse_data(data)

    # Create polynomial features of the two features up to degree 5
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)

    feature_names = poly.get_feature_names_out(['X1', 'X2'])

    C_values = [0.1, 1, 10, 100, 1000]

    for C in C_values:
        print(f"\n{model_type} Regression with C = {C}")

        if model_type == "Lasso":
            model = Lasso(alpha=1 / (2 * C), max_iter=10000)
        else:
            model = Ridge(alpha=1 / (2 * C), max_iter=10000)

        model.fit(X_poly, y)

        # Report non-zero parameters (for Ridge, report parameters above a small threshold)
        threshold = 1e-4 if model_type == "Ridge" else 0
        significant_params = [(name, coef) for name, coef in zip(feature_names, model.coef_)
                              if abs(coef) > threshold]

        if significant_params:
            print("Significant parameters:")
            for name, coef in significant_params:
                print(f"  {name}: {coef:.6f}")
        else:
            print("All parameters are zero.")


def visualize_predictions(model_type: str, data_path: str) -> None:
    """
    Create prediction surface visualizations for either Lasso or Ridge regression
    """
    data = read_data(data_path)
    X, y = parse_data(data)

    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)

    C_values = [0.1, 1, 10, 100, 1000]

    grid = np.linspace(-5, 5, 50)
    xx, yy = np.meshgrid(grid, grid)
    Xtest = np.column_stack((xx.ravel(), yy.ravel()))
    Xtest_poly = poly.transform(Xtest)

    for C in C_values:
        if model_type == "Lasso":
            model = Lasso(alpha=1 / (2 * C), max_iter=10000)
        else:
            model = Ridge(alpha=1 / (2 * C), max_iter=10000)

        model.fit(X_poly, y)
        y_pred = model.predict(Xtest_poly)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(xx, yy, y_pred.reshape(xx.shape), cmap='viridis', alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], y, c='r', marker='o')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        ax.set_title(f'{model_type} Regression Predictions (C = {C})')

        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.legend(['Predictions', 'Training Data'])

        plt.savefig(os.path.join(GRAPHS_PATH, f'{model_type.lower()}_predictions_C_{C}.png'))
        plt.close()
        print(f"{model_type} plot for C = {C} saved.")


def compare_models() -> None:
    """
    Part v
    Compare Lasso and Ridge regression models
    """
    print("\nAnalyzing Lasso Regression:")
    train_and_analyze_model("Lasso", DATA_PATH)
    print("\nVisualizing Lasso Predictions:")
    visualize_predictions("Lasso", DATA_PATH)

    print("\nAnalyzing Ridge Regression:")
    train_and_analyze_model("Ridge", DATA_PATH)
    print("\nVisualizing Ridge Predictions:")
    visualize_predictions("Ridge", DATA_PATH)


def cross_validate_models() -> None:
    """
    Perform 5-fold cross-validation for both Lasso and Ridge regression,
    plot error bars for different C values, and print the optimal C value
    for each model.
    """
    # Load and prepare data
    data = read_data(DATA_PATH)
    X, y = parse_data(data)

    # Create polynomial features
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Define C values on a logarithmic scale
    # We choose a wider range than before to better understand the error patterns
    # Starting from 0.01 to capture high regularization effects
    # Up to 10000 to see if the error stabilizes or starts increasing
    C_values = np.logspace(-2, 4, 20)  # 20 points from 10^-2 to 10^4

    # Initialize arrays to store results
    models = {
        "Lasso": Lasso,
        "Ridge": Ridge
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model_class in models.items():
        errors = []

        for C in C_values:
            fold_errors = []

            for train_idx, val_idx in kf.split(X_poly):
                # Split data
                X_train, X_val = X_poly[train_idx], X_poly[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train model
                model = model_class(alpha=1 / (2 * C), max_iter=10000)
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                fold_errors.append(mse)

            errors.append(fold_errors)

        # Convert to numpy array for easier calculations
        errors = np.array(errors)
        mean_errors = np.mean(errors, axis=1)
        std_errors = np.std(errors, axis=1)

        # Find optimal C value
        best_idx = np.argmin(mean_errors)
        best_C = C_values[best_idx]
        best_error = mean_errors[best_idx]
        best_std = std_errors[best_idx]

        # Print results
        print(f"\n{model_name} Regression Results:")
        print(f"Optimal C value: {best_C:.2e}")
        print(f"Mean squared error at optimal C: {best_error:.2e} ± {best_std:.2e}")

        # Plot error bars
        plt.figure(figsize=(10, 6))
        plt.errorbar(C_values, mean_errors, yerr=std_errors, fmt='o-', capsize=5,
                     label=f'{model_name} Regression')

        # Add marker for optimal point
        plt.plot(best_C, best_error, 'r*', markersize=15,
                label=f'Optimal C = {best_C:.2e}')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('C (log scale)')
        plt.ylabel('Mean Squared Error (log scale)')
        plt.title(f'5-Fold Cross-Validation Error vs C\n{model_name} Regression')
        plt.grid(True)
        plt.legend()

        # Save plot
        plt.savefig(os.path.join(GRAPHS_PATH, f'{model_name.lower()}_cv_error.png'))
        plt.close()
        print(f"{model_name} cross-validation plot saved.")


def main():
    # Original parts
    # part_i()
    # part_ii(DATA_PATH)
    # part_iii(DATA_PATH)

    # New comparison analysis
    # compare_models()

    cross_validate_models()


if __name__ == "__main__":
    main()