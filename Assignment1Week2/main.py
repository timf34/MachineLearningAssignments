import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

DATA_PATH: str = "./data/week2.csv"


def read_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def parse_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    X1 = df.iloc[:, 0]
    X2 = df.iloc[:, 1]
    y = df.iloc[:, 2]
    return X1, X2, y


def plot_data(X1: np.ndarray, X2: np.ndarray, y: np.ndarray, markers: dict, title: str, file_name: str) -> None:
    plt.figure()
    for xi1, xi2, yi in zip(X1, X2, y):
        label = f"Class {yi}" if f"Class {yi}" not in plt.gca().get_legend_handles_labels()[1] else ""
        plt.plot(xi1, xi2, markers[yi]['marker'], color=markers[yi]['color'], label=label)

    plt.xlabel('Feature 1 (x_1)')
    plt.ylabel('Feature 2 (x_2)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)


def visualize_data(X1: np.ndarray, X2: np.ndarray, y: np.ndarray) -> None:
    markers = {
        1: {'marker': '+', 'color': 'blue'},
        -1: {'marker': '+', 'color': 'green'}
    }
    plot_data(X1, X2, y, markers, 'Training Data', 'training_data_plot.png')


def visualize_predictions(X1: np.ndarray, X2: np.ndarray, y: np.ndarray, predictions: np.ndarray, clf: LogisticRegression) -> None:
    plt.figure()

    # Original training data markers
    markers = {
        1: {'marker': '+', 'color': 'blue'},
        -1: {'marker': 'o', 'color': 'red'}
    }
    plot_data(X1, X2, y, markers, 'Training Data and Predictions', 'training_data_with_predictions.png')

    # Predicted data markers
    pred_markers = {
        1: {'marker': 'x', 'color': 'green'},
        -1: {'marker': 's', 'color': 'orange'}
    }
    plot_data(X1, X2, predictions, pred_markers, '', '')

    # Plot decision boundary
    xmin, xmax = X1.min() - 0.1, X1.max() + 0.1
    ymin, ymax = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, probs, levels=[0.5], linewidths=2, colors='black', linestyles='dashed')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_data_with_predictions.png')


def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf


def main():
    df = read_data(DATA_PATH)
    X1, X2, y = parse_data(df)
    X = np.column_stack((X1, X2))

    # (a)(i) Visualize the data
    visualize_data(X1, X2, y)

    # (a)(ii) Train logistic regression model
    clf = train_logistic_regression(X, y)
    predictions = clf.predict(X)
    parameters = np.concatenate((clf.intercept_, clf.coef_.flatten()))
    print(f"Model parameters: {parameters}")
    print(f"Intercept: {clf.intercept_[0]}")
    print(f"Coefficients: {clf.coef_[0]}")

    # Discuss feature influence
    coef = clf.coef_[0]
    most_influential_feature = 'Feature 1' if abs(coef[0]) > abs(coef[1]) else 'Feature 2'
    print(f"The most influential feature is {most_influential_feature}")
    print(f"Feature 1 coefficient ({coef[0]}): {'increases' if coef[0]>0 else 'decreases'} the prediction")
    print(f"Feature 2 coefficient ({coef[1]}): {'increases' if coef[1]>0 else 'decreases'} the prediction")

    # (a)(iii) Add predictions to the plot
    visualize_predictions(X1, X2, y, predictions, clf)

    # (a)(iv) Comment on predictions vs training data
    correct = (predictions == y).sum()
    total = len(y)
    accuracy = correct / total
    print(f"Accuracy on training data: {accuracy*100:.2f}%")
    print("The model predictions match the training data well.")

if __name__ == '__main__':
    main()
