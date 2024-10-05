import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

DATA_PATH: str = "./data/week2.csv"


def read_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def parse_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    X1 = df.iloc[:, 0]
    X2 = df.iloc[:, 1]
    y = df.iloc[:, 2]
    return X1, X2, y


def plot_data(X1: np.ndarray, X2: np.ndarray, y: np.ndarray, markers: dict, title: str, file_name: str) -> None:
    plt.figure(figsize=(8, 6))  # Increase figure size for better layout
    for xi1, xi2, yi in zip(X1, X2, y):
        label = f"Class {yi}" if f"Class {yi}" not in plt.gca().get_legend_handles_labels()[1] else ""
        plt.plot(xi1, xi2, markers[yi]['marker'], color=markers[yi]['color'], label=label)

    plt.xlabel('Feature 1 (x_1)')
    plt.ylabel('Feature 2 (x_2)')
    plt.title(title)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    plt.grid(True)

    plt.tight_layout()  # Adjust the layout so everything fits well
    plt.savefig(file_name, bbox_inches='tight')

def visualize_data(X1: np.ndarray, X2: np.ndarray, y: np.ndarray) -> None:
    markers = {
        1: {'marker': '+', 'color': 'green'},
        -1: {'marker': 'o', 'color': 'blue'}
    }
    plot_data(X1, X2, y, markers, 'Training Data', 'training_data_plot.png')


def visualize_linear_classifier_predictions(X1: np.ndarray, X2: np.ndarray, y: np.ndarray, predictions: np.ndarray, clf: LogisticRegression, filename: str) -> None:
    plt.figure()

    # Original data markers
    markers = {
        1: {'marker': 'o', 'color': 'green', 'label': 'Actual Class 1'},
        -1: {'marker': 'o', 'color': 'blue', 'label': 'Actual Class -1'}
    }
    for xi1, xi2, yi in zip(X1, X2, y):
        plt.plot(xi1, xi2, markers[yi]['marker'], color=markers[yi]['color'], label=markers[yi]['label'] if markers[yi]['label'] not in plt.gca().get_legend_handles_labels()[1] else "")

    # Predicted data markers
    pred_markers = {
        1: {'marker': 'x', 'color': 'red', 'label': 'Predicted Class 1'},
        -1: {'marker': 's', 'color': 'purple', 'label': 'Predicted Class -1'}
    }
    for xi1, xi2, pred in zip(X1, X2, predictions):
        plt.plot(xi1, xi2, pred_markers[pred]['marker'], color=pred_markers[pred]['color'], label=pred_markers[pred]['label'] if pred_markers[pred]['label'] not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot decision boundary
    xmin, xmax = X1.min() - 0.1, X1.max() + 0.1
    ymin, ymax = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    plt.contour(xx, yy, probs, levels=[0.5], linewidths=2, colors='black', linestyles='dashed')


    plt.xlabel('Feature 1 (x_1)')
    plt.ylabel('Feature 2 (x_2)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    plt.grid(True)
    plt.title('Training Data with Predictions and Decision Boundary')
    plt.savefig(filename, bbox_inches='tight')


def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf


def train_svm(X: np.ndarray, y: np.ndarray, C: float) -> LinearSVC:
    clf = LinearSVC(C=C, max_iter=10000)
    clf.fit(X, y)
    return clf


def visualize_svm_predictions(X1: np.ndarray, X2: np.ndarray, y: np.ndarray, predictions: np.ndarray, clf: LinearSVC, filename: str) -> None:
    plt.figure(figsize=(10, 8))

    # Original data markers
    markers = {
        1: {'marker': 'o', 'color': 'green', 'label': 'Actual Class 1'},
        -1: {'marker': 'o', 'color': 'blue', 'label': 'Actual Class -1'}
    }
    for xi1, xi2, yi in zip(X1, X2, y):
        plt.plot(xi1, xi2, markers[yi]['marker'], color=markers[yi]['color'], label=markers[yi]['label'] if markers[yi]['label'] not in plt.gca().get_legend_handles_labels()[1] else "")

    # Predicted data markers
    pred_markers = {
        1: {'marker': 'x', 'color': 'red', 'label': 'Predicted Class 1'},
        -1: {'marker': 'x', 'color': 'yellow', 'label': 'Predicted Class -1'}
    }
    for xi1, xi2, pred in zip(X1, X2, predictions):
        plt.plot(xi1, xi2, pred_markers[pred]['marker'], color=pred_markers[pred]['color'], label=pred_markers[pred]['label'] if pred_markers[pred]['label'] not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot decision boundary
    xmin, xmax = X1.min() - 0.1, X1.max() + 0.1
    ymin, ymax = X2.min() - 0.1, X2.max() + 0.1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='dashed')

    plt.xlabel('Feature 1 (x_1)')
    plt.ylabel('Feature 2 (x_2)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    plt.grid(True)
    plt.title('SVM: Training Data with Predictions and Decision Boundary')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def part_a(X, X1, X2, y):
    # ########### Part A ############

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
    visualize_linear_classifier_predictions(X1, X2, y, predictions, clf, "visualize_linear_classifier_predictions.png")

    # (a)(iv) Comment on predictions vs training data

    report = classification_report(y, predictions)
    conf_matrix = confusion_matrix(y, predictions)
    print(f"Classification report:\n{report}")
    print(f"\nConfusion matrix:\n{conf_matrix}")

    correct = (predictions == y).sum()
    total = len(y)
    accuracy = correct / total
    print(f"Accuracy on training data: {accuracy*100:.2f}%")


def part_b(X, X1, X2, y):
    # Train and analyze SVMs for different values of C
    C_values = [0.001, 1, 100]
    for C in C_values:
        print(f"Training SVM with C={C}")
        clf = train_svm(X, y, C)
        predictions = clf.predict(X)

        # Print model parameters
        parameters = np.concatenate((clf.intercept_, clf.coef_.flatten()))
        print(f"Model parameters for SVM with C={C}: {parameters}")
        print(f"Intercept: {clf.intercept_[0]}")
        print(f"Coefficients: {clf.coef_[0]}")

        # Accuracy
        accuracy = (predictions == y).mean() * 100
        print(f"Accuracy for C={C}: {accuracy:.2f}%")

        # Other metrics
        report = classification_report(y, predictions)
        conf_matrix = confusion_matrix(y, predictions)
        print(f"Classification report:\n{report}")
        print(f"\nConfusion matrix:\n{conf_matrix}")

        # Visualise the predictions on top of the base data
        visualize_svm_predictions(X1, X2, y, predictions, clf, f"svm_predictions_C_{C}.png")

        print("\n\n ------------------- \n\n")


def main():
    df = read_data(DATA_PATH)
    X1, X2, y = parse_data(df)
    X = np.column_stack((X1, X2))

    # Part A
    # part_a(X, X1, X2, y)

    # Part B
    part_b(X, X1, X2, y)


if __name__ == '__main__':
    main()
