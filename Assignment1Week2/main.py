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


def visualize_data(X1: np.ndarray, X2: np.ndarray, y: np.ndarray) -> None:
    plt.figure()
    for xi1, xi2, yi in zip(X1, X2, y):
        if yi == 1:
            plt.plot(xi1, xi2, '+', color='blue', label='Class +1' if 'Class +1' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.plot(xi1, xi2, '+', color='green', label='Class -1' if 'Class -1' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel('Feature 1 (x_1')
    plt.ylabel('Feature 2 (x_2)')
    plt.title('Training Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_data_plot.png')


def main():
    df = read_data(DATA_PATH)
    X1, X2, y = parse_data(df)
    X = np.column_stack((X1, X2))

    # (a)(i) Visualize the data
    visualize_data(X1, X2, y)

    # (a)(ii) Train logistic regression model
    clf = LogisticRegression()
    clf.fit(X, y)
    predictions = clf.predict(X)
    parameters = np.concatenate((clf.intercept_, clf.coef_.flatten()))
    print("Model parameters:")
    print(f"Intercept: {clf.intercept_[0]}")
    print(f"Coefficients: {clf.coef_[0]}")

    # Discuss feature influence
    coef = clf.coef_[0]
    if abs(coef[0]) > abs(coef[1]):
        most_influential_feature = 'Feature 1'
    else:
        most_influential_feature = 'Feature 2'
    print(f"The most influential feature is {most_influential_feature}")

    print(f"Feature 1 coefficient ({coef[0]}): {'increases' if coef[0]>0 else 'decreases'} the prediction")
    print(f"Feature 2 coefficient ({coef[1]}): {'increases' if coef[1]>0 else 'decreases'} the prediction")

    # (a)(iii) Add predictions to the plot
    plt.figure()
    # Plot the original data
    for xi1, xi2, yi in zip(X1, X2, y):
        if yi == 1:
            plt.plot(xi1, xi2, '+', color='blue', label='Class +1' if 'Class +1' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.plot(xi1, xi2, 'o', color='red', label='Class -1' if 'Class -1' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot predictions
    for xi1, xi2, pi in zip(X1, X2, predictions):
        if pi == 1:
            plt.plot(xi1, xi2, 'x', color='green', label='Predicted +1' if 'Predicted +1' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.plot(xi1, xi2, 's', color='orange', label='Predicted -1' if 'Predicted -1' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot decision boundary
    xmin, xmax = X1.min() - 0.1, X1.max() + 0.1
    ymin, ymax = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, probs, levels=[0.5], linewidths=2, colors='black', linestyles='dashed')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data and Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_data_with_predictions.png')

    # (a)(iv) Comment on predictions vs training data
    correct = (predictions == y).sum()
    total = len(y)
    accuracy = correct / total
    print(f"Accuracy on training data: {accuracy*100:.2f}%")
    print("The model predictions match the training data well.")


if __name__ == '__main__':
    main()
