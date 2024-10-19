import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
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


def ib() -> None:
    """
    Train a k-Nearest Neighbors (kNN) classifier on the data, using cross-validation
    to select the optimal value for k. Provide data, explanations, and analysis to
    justify the choice.
    """
    # Read and parse data
    df = read_data(DATA_PATH)
    X, y = parse_data(df)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define the parameter grid for k
    param_grid = {'n_neighbors': range(1, 31)}  # Test k values from 1 to 30

    # Set up k-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Create and train the kNN classifier using GridSearchCV
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_scaled, y)

    # Get the best model and its parameters
    best_k = grid_search.best_params_['n_neighbors']
    best_model = grid_search.best_estimator_

    # Print results
    print(f"Best k value: {best_k}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Evaluate the model on the entire dataset
    y_pred = best_model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    print(f"Final model accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot cross-validation scores for different k values
    cv_results = grid_search.cv_results_
    k_values = cv_results['param_n_neighbors'].data
    mean_scores = cv_results['mean_test_score']
    std_scores = cv_results['std_test_score']

    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, mean_scores, yerr=std_scores, fmt='o-')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Cross-validation Accuracy')
    plt.title('kNN: Cross-validation Accuracy vs. k')
    plt.savefig(f"{GRAPHS_PATH}knn_cv_scores.png")
    plt.close()

    # Plot decision boundary
    plot_decision_boundary(X_scaled, y, best_model, 'kNN Decision Boundary')

    # Analysis and explanation
    print("\nAnalysis and Explanation:")
    print(f"1. The optimal k value of {best_k} was selected based on cross-validation accuracy.")
    print("2. This k value balances between overfitting (low k) and underfitting (high k).")
    print("3. The cross-validation plot shows how accuracy varies with different k values.")
    print("4. The decision boundary plot illustrates how the model separates classes in feature space.")
    print("5. kNN can capture nonlinear decision boundaries without need for polynomial features.")
    print("6. The confusion matrix shows the model's performance across different classes.")
    print("\nNote: The exact optimal k may vary slightly due to random data splits in cross-validation.")


def train_most_frequent_classifier(X: np.ndarray, y: np.ndarray) -> DummyClassifier:
    """Train a classifier that always predicts the most frequent class."""
    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(X, y)
    return clf


def train_random_classifier(X: np.ndarray, y: np.ndarray) -> DummyClassifier:
    """Train a classifier that makes random predictions."""
    clf = DummyClassifier(strategy='uniform')
    clf.fit(X, y)
    return clf


def plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
    """Plot a confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(f"{GRAPHS_PATH}{title.lower().replace(' ', '_')}.png")
    plt.close()


def plot_roc_curves(X: np.ndarray, y: np.ndarray) -> None:
    """
    Plot ROC curves for Logistic Regression and kNN classifiers,
    and include points for baseline classifiers.
    """
    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train classifiers
    lr_model, _ = train_logistic_regression_cv(X, y)
    knn_model = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, 31)}, cv=5, n_jobs=-1)
    knn_model.fit(X_scaled, y)
    mf_model = train_most_frequent_classifier(X, y)
    rand_model = train_random_classifier(X, y)

    # Compute ROC curve and ROC area for each classifier
    fpr_lr, tpr_lr, _ = roc_curve(y, lr_model.predict_proba(X)[:, 1])
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    fpr_knn, tpr_knn, _ = roc_curve(y, knn_model.predict_proba(X_scaled)[:, 1])
    roc_auc_knn = auc(fpr_knn, tpr_knn)

    # Compute points for baseline classifiers
    fpr_mf, tpr_mf, _ = roc_curve(y, mf_model.predict(X))
    fpr_rand, tpr_rand, _ = roc_curve(y, rand_model.predict(X))

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2,
             label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    plt.plot(fpr_knn, tpr_knn, color='green', lw=2,
             label=f'kNN (AUC = {roc_auc_knn:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Guess')
    plt.scatter(fpr_mf[1], tpr_mf[1], color='red', marker='o', s=100,
                label='Most Frequent')
    plt.scatter(fpr_rand[1], tpr_rand[1], color='blue', marker='s', s=100,
                label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{GRAPHS_PATH}roc_curves.png")
    plt.close()


def compare_classifiers():
    """Compare Logistic Regression, kNN, and baseline classifiers using confusion matrices."""
    # Read and parse data
    df = read_data(DATA_PATH)
    X, y = parse_data(df)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Logistic Regression
    lr_model, _ = train_logistic_regression_cv(X, y)
    y_pred_lr = lr_model.predict(X)
    cm_lr = confusion_matrix(y, y_pred_lr)

    # Train kNN
    knn_model = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, 31)}, cv=5, n_jobs=-1)
    knn_model.fit(X_scaled, y)
    y_pred_knn = knn_model.predict(X_scaled)
    cm_knn = confusion_matrix(y, y_pred_knn)

    # Train Most Frequent Classifier
    mf_model = train_most_frequent_classifier(X, y)
    y_pred_mf = mf_model.predict(X)
    cm_mf = confusion_matrix(y, y_pred_mf)

    # Train Random Classifier
    rand_model = train_random_classifier(X, y)
    y_pred_rand = rand_model.predict(X)
    cm_rand = confusion_matrix(y, y_pred_rand)

    # Plot confusion matrices
    plot_confusion_matrix(cm_lr, "Logistic Regression Confusion Matrix")
    plot_confusion_matrix(cm_knn, "kNN Confusion Matrix")
    plot_confusion_matrix(cm_mf, "Most Frequent Classifier Confusion Matrix")
    plot_confusion_matrix(cm_rand, "Random Classifier Confusion Matrix")

    # Print accuracies
    print(f"Logistic Regression Accuracy: {accuracy_score(y, y_pred_lr):.4f}")
    print(f"kNN Accuracy: {accuracy_score(y, y_pred_knn):.4f}")
    print(f"Most Frequent Classifier Accuracy: {accuracy_score(y, y_pred_mf):.4f}")
    print(f"Random Classifier Accuracy: {accuracy_score(y, y_pred_rand):.4f}")

    # Plot ROC curves
    plot_roc_curves(X, y)


if __name__ == "__main__":
    # ia()
    # ib()
    compare_classifiers()
