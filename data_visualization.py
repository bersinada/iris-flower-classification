import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def create_visualizations():
    """Create comprehensive visualizations for the Iris dataset"""

    # Load data
    iris= load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Set style
    fig, axes = plt.subplots(2, 2, figsize=(15, 30))

    # 1. Sepal Length vs Sepal Width
    for i, species in enumerate(df['species'].unique()):
        subset = df[df['species'] == species]
        axes[0, 0].scatter(subset['sepal length (cm)'], 
                           subset['sepal width (cm)'],
                           label=species, alpha=0.7, s=60)
    
    axes[0, 0].set_title('Sepal Length vs Sepal Width')
    axes[0, 0].set_xlabel('Sepal Length (cm)')
    axes[0, 0].set_ylabel('Sepal Width (cm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Petal Length vs Petal Width
    for i, species in enumerate(df['species'].unique()):
        subset = df[df['species'] == species]
        axes[0, 1].scatter(subset['petal length (cm)'], 
                           subset['petal width (cm)'],
                           label=species, alpha=0.7, s=60)
                           
    axes[0, 1].set_title('Petal Length vs Petal Width')
    axes[0, 1].set_xlabel('Petal Length (cm)')
    axes[0, 1].set_ylabel('Petal Width (cm)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Feature distributions
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    for i, feature in enumerate(features):
        axes[1,0].hist(df[feature], bins=20, alpha=0.7, label=feature, edgecolor='black')
    axes[1,0].set_title('Feature Distributions')
    axes[1,0].legend()

    # 4. Correlation heatmap
    correlation_matrix = df.drop('species', axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1,1], square=True, fmt='.2f')
    axes[1, 1].set_title('Correlation Matrix')

    plt.tight_layout()
    plt.savefig('iris_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Viusalizations created and saved as 'iris_analysis.png'")

    # Additional insights
    print("\n=== KEY INSIGHTS ===")
    print("1. Setosa is clearly separable from other species")
    print("2. Versicolor and Virginica have some overlap")
    print("3. Petal measurements are more discriminative than sepal measurements")
    print("4. Strong correlation between petal length and petal width")

if __name__ == "__main__":
    create_visualizations()