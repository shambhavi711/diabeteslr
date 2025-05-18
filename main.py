import os
from data_preprocessing import preprocess_pipeline
from diabetes_model import (
    LogisticRegression,
    compute_metrics,
    plot_cost_history,
    plot_confusion_matrix,
    plot_roc_curve
)

def main():
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Preprocess the data
    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaling_params = preprocess_pipeline(
        random_state=42
    )
    
    # Initialize and train the model
    print("\nTraining model...")
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    cost_history = model.fit(X_train, y_train, X_val, y_val, patience=20)
    
    # Save model weights
    model.save_weights('model_weights.npz')
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_cost_history(cost_history, save_path='plots/cost_history.png')
    plot_confusion_matrix(metrics['confusion_matrix'], save_path='plots/confusion_matrix.png')
    plot_roc_curve(y_test, y_pred_proba, save_path='plots/roc_curve.png')
    
    print("\nTraining complete! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main() 