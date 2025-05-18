import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import urllib.request
import os

def download_dataset(url: str = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv",
                    save_path: str = "diabetes.csv") -> None:
    """Download the Pima Indians Diabetes dataset."""
    if not os.path.exists(save_path):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print("Download complete!")

def load_data(filepath: str = "diabetes.csv") -> pd.DataFrame:
    """Load the dataset into a pandas DataFrame."""
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    return pd.read_csv(filepath, names=column_names)

def handle_missing_values(df: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        method: 'mean' or 'median' for imputation
    
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    
    # Replace zeros with NaN for columns where 0 is not a valid value
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_columns] = df[zero_columns].replace(0, np.nan)
    
    # Impute missing values
    if method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'median':
        df = df.fillna(df.median())
    else:
        raise ValueError("Method must be either 'mean' or 'median'")
    
    return df

def normalize_features(df: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, dict]:
    """
    Normalize features using either Min-Max scaling or Standard scaling.
    
    Args:
        df: Input DataFrame
        method: 'minmax' or 'standard' scaling
    
    Returns:
        Tuple of (normalized DataFrame, scaling parameters)
    """
    df = df.copy()
    scaling_params = {}
    
    # Don't normalize the target variable
    feature_columns = df.columns[:-1]
    
    for column in feature_columns:
        if method == 'minmax':
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
            scaling_params[column] = {'min': min_val, 'max': max_val}
        elif method == 'standard':
            mean_val = df[column].mean()
            std_val = df[column].std()
            df[column] = (df[column] - mean_val) / (std_val + 1e-8)  # Add small epsilon to prevent division by zero
            scaling_params[column] = {'mean': mean_val, 'std': std_val}
        else:
            raise ValueError("Method must be either 'minmax' or 'standard'")
    
    return df, scaling_params

def plot_feature_correlations(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot correlation heatmap of features."""
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def split_data(df: pd.DataFrame, test_size: float = 0.2, 
               val_size: float = 0.2, random_state: Optional[int] = None) -> Tuple:
    """
    Split data into training, validation, and test sets.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state)
    
    # Calculate split indices
    n_samples = len(df)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # Split the data
    train_data = df.iloc[:val_idx]
    val_data = df.iloc[val_idx:test_idx]
    test_data = df.iloc[test_idx:]
    
    # Separate features and target
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    
    X_val = val_data.iloc[:, :-1].values
    y_val = val_data.iloc[:, -1].values
    
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_pipeline(filepath: str = "diabetes.csv", 
                       imputation_method: str = 'median',
                       scaling_method: str = 'standard',
                       test_size: float = 0.2,
                       val_size: float = 0.2,
                       random_state: Optional[int] = None) -> Tuple:
    """
    Complete preprocessing pipeline for the diabetes dataset.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaling_params)
    """
    # Download dataset if not exists
    download_dataset(save_path=filepath)
    
    # Load and preprocess data
    df = load_data(filepath)
    df = handle_missing_values(df, method=imputation_method)
    df, scaling_params = normalize_features(df, method=scaling_method)
    
    # Plot correlations
    plot_feature_correlations(df, save_path='correlation_heatmap.png')
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, test_size, val_size, random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaling_params 
