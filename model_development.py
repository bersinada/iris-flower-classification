import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    """Prepare data for modeling"""

    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=== DATA PREPARATION ===")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
    }

    results = {}
    
    print("\n=== MODEL TRAINING ===")

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'predictions': y_pred
        }

        print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test accuracy: {accuracy:.4f}")
        
    return results

def evaluate_models(results, y_test):
    """Evaluate and compare model perforamnce"""

    print("\n=== MODEL COMPARISON ===")

    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
    best_model = results[best_model_name]['model']

    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {results[best_model_name]['test_accuracy']:.4f}")

    # Detailed evaluation of best model
    y_pred = results[best_model_name]['predictions']

    print(f"\n=== {best_model_name} DETAILED EVALUATION ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                            target_names=['setosa', 'versicolor', 'virginica']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['setosa', 'versicolor', 'virginica'],
                yticklabels=['setosa', 'versicolor', 'virginica'])
    plt.title(f'{best_model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return best_model, best_model_name

def main():
    """Main function"""
    print("=== IRIS FLOWER CLASSIFICATION - MODEL DEVELOPMENT ===\n")

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Evaluate models
    best_model, best_model_name = evaluate_models(results, y_test)

    print(f"\nModel development completed:")
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {results[best_model_name]['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()