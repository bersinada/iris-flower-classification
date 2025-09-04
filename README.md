# Iris Flower Classification

**Practice Project** - Hands-on learning of data science concepts through the classic Iris flower classification problem.

## 🎯 Live Demo

**🌐 [Try the Live Application](https://irisflowerclassificationproject.streamlit.app/)**

## 📚 What This Project Covers

- Data exploration and analysis (EDA)
- Data visualization techniques
- Machine learning model development
- Model evaluation and comparison
- Web application development
- Deployment

## 🎯 About the Dataset

The Iris dataset contains 150 samples with 4 features:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

And 3 classes: setosa, versicolor, virginica

## 📁 Project Files

- `requirements.txt` - Python packages
- `data_exploration.py` - Initial data analysis
- `data_visualization.py` - Data visualizations
- `model_development.py` - ML model training and comparison
- `app.py` - Streamlit web application
- `iris_model.pkl` - Trained SVM model
- `iris_scaler.pkl` - Feature scaler
- `iris_analysis.png` - Generated plots
- `confusion_matrix.png` - Model performance visualization

## 🚀 Getting Started

### Local Development

```bash
pip install -r requirements.txt
python data_exploration.py
python data_visualization.py
python model_development.py
streamlit run app.py
```

### Web Application

Visit: https://irisflowerclassificationproject.streamlit.app/

## 🤖 Model Performance

- **Best Model**: Support Vector Machine (SVM)
- **Accuracy**: 96.67%
- **Cross-validation**: 96.67% (±3.12%)

### Class-wise Performance:

- **Setosa**: 100% precision, 100% recall
- **Versicolor**: 100% precision, 90% recall
- **Virginica**: 91% precision, 100% recall

## 🌐 Web Application Features

### 📄 Pages:

1. **Home** - Project overview and dataset summary
2. **Data Analysis** - Interactive visualizations and statistics
3. **Prediction** - Real-time Iris species prediction
4. **Model Performance** - Detailed model metrics and evaluation

### 🎮 Interactive Features:

- Real-time predictions with sliders
- Interactive plots and visualizations
- Example values for each species
- Probability scores for predictions

## 🛠️ Technologies Used

- **Python** - Main programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Data visualization
- **Plotly** - Interactive visualizations
- **Streamlit** - Web application framework
- **Joblib** - Model serialization

## 🚀 Deployment

This project is deployed on **Streamlit Cloud** and automatically updates when changes are pushed to GitHub.

## 📝 Future Improvements

- Add more machine learning algorithms
- Implement hyperparameter tuning
- Add data upload functionality
- Create mobile-responsive design
- Add model explanation features

---

*This is a practice project for learning data science concepts hands-on.*
