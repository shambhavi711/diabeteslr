import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
import os

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.001, n_iterations: int = 1000, lambda_reg: float = 0.1):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            n_iterations (int): Maximum number of iterations for training
            lambda_reg (float): Regularization parameter
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
    
    def initialize_parameters(self, n_features: int):
        """Initialize weights and bias using Xavier/Glorot initialization."""
        self.weights = np.random.randn(n_features) * np.sqrt(2.0 / n_features)
        self.bias = 0
        
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute binary cross-entropy loss with L2 regularization."""
        m = X.shape[0]
        predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
        cost = -(1/m) * np.sum(y * np.log(predictions + 1e-15) + 
                              (1 - y) * np.log(1 - predictions + 1e-15))
        # Add L2 regularization
        cost += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        return cost
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients for weights and bias with L2 regularization."""
        m = X.shape[0]
        predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
        
        dw = (1/m) * np.dot(X.T, (predictions - y)) + (self.lambda_reg / m) * self.weights
        db = (1/m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, patience: int = 20) -> List[float]:
        """
        Train the logistic regression model using gradient descent with momentum.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            patience: Number of iterations to wait for improvement before early stopping
            
        Returns:
            List of cost values during training
        """
        self.initialize_parameters(X.shape[1])
        best_val_cost = float('inf')
        patience_counter = 0
        momentum_w = np.zeros_like(self.weights)
        momentum_b = 0
        beta = 0.9  # Momentum parameter
        
        for i in range(self.n_iterations):
            # Compute gradients
            dw, db = self.compute_gradients(X, y)
            
            # Update with momentum
            momentum_w = beta * momentum_w + (1 - beta) * dw
            momentum_b = beta * momentum_b + (1 - beta) * db
            
            # Update parameters
            self.weights -= self.learning_rate * momentum_w
            self.bias -= self.learning_rate * momentum_b
            
            # Compute and store cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Early stopping with validation set
            if X_val is not None and y_val is not None:
                val_cost = self.compute_cost(X_val, y_val)
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {i}")
                    break
            
            if i % 100 == 0:
                print(f"Cost at iteration {i}: {cost:.4f}")
                
        return self.cost_history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions using the trained model."""
        predictions = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return (predictions >= threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability predictions."""
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def save_weights(self, filepath: str):
        """Save model weights and bias to a file."""
        np.savez(filepath, weights=self.weights, bias=self.bias)
    
    def load_weights(self, filepath: str):
        """Load model weights and bias from a file."""
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics.
    
    Returns:
        Dictionary containing accuracy, precision, recall, and confusion matrix
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': confusion_matrix
    }

def plot_cost_history(cost_history: List[float], save_path: Optional[str] = None):
    """Plot the cost history during training."""
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history)
    plt.title('Cost vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(confusion_matrix: np.ndarray, save_path: Optional[str] = None):
    """Plot the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: Optional[str] = None):
    """Plot the ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
