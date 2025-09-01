import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def load_and_explore_data():
    """Load Iris dataset and display basic information"""
    
    # Load the dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("=== IRIS DATASET INFORMATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {iris.feature_names}")
    print(f"Classes: {iris.target_names}")
    print("\nClass distribution:")
    print(df['species'].value_counts())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    return df, iris

if __name__ == "__main__":
    df, iris = load_and_explore_data() 