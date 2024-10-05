import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

# Example usage
DATA_PATH = "./data/week3.csv"
GRAPHS_PATH = "./graphs/"


def read_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def parse_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Extract features (assuming all columns except the last are features)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values  # Assuming the last column is the target
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


def main():
    # Part I
    part_i()


if __name__ == "__main__":
    main()