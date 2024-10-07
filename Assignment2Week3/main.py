import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
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


def create_3d_scatter_plot(data_path: str, save_path: str, filename: str, x_label: str = 'Feature 1', y_label: str = 'Feature 2', z_label: str = 'Target') -> None:
    """
    Create a 3D scatter plot from a CSV file and save it to the specified path.

    Parameters:
    - data_path (str): Path to the CSV file containing the dataset.
    - save_path (str): Path where the 3D scatter plot image will be saved.
    - filename (str): Name of the file to save the plot as.
    - x_label (str): Label for the x-axis (default 'Feature 1').
    - y_label (str): Label for the y-axis (default 'Feature 2').
    - z_label (str): Label for the z-axis (default 'Target').
    """
    # Load the dataset
    data = read_data(data_path)

    X, y = parse_data(data)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with the first feature on the x-axis, second feature on the y-axis, and target on the z-axis
    ax.scatter(X[:, 0], X[:, 1], y)

    # Label the axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    # Save the plot to the specified path
    save_path = os.path.join(save_path, filename)
    plt.savefig(save_path)

    print(f"3D scatter plot saved to {save_path}")


def part_i() -> None:
    # part a
    create_3d_scatter_plot(data_path=DATA_PATH, save_path=GRAPHS_PATH, filename="3d_scatter_plot.png")


def part_ii(data_path: str) -> None:
    # Load and parse data
    data = read_data(data_path)
    X, y = parse_data(data)

    # Create polynomial features of the two features up to degree 5
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Get feature names
    feature_names = poly.get_feature_names_out(['X1', 'X2'])

    # Define a range of C values
    C_values = [0.1, 1, 10, 100, 1000]

    for C in C_values:
        print(f"\nLasso Regression with C = {C}")

        # Train Lasso model
        lasso = Lasso(alpha=1 / (2 * C), max_iter=10000)
        lasso.fit(X_poly, y)

        # Report non-zero parameters
        non_zero_params = [(name, coef) for name, coef in zip(feature_names, lasso.coef_) if coef != 0]

        if non_zero_params:
            print("Non-zero parameters:")
            for name, coef in non_zero_params:
                print(f"  {name}: {coef:.6f}")
        else:
            print("All parameters are zero.")


def main():
    part_ii(DATA_PATH)


if __name__ == "__main__":
    main()