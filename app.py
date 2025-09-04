import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon=":flower:",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #222831;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2B2B2B;
    }
    .prediction-box {
        background-color: #222831;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #2B2B2B;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

@st.cache_resource
def load_or_train_model():
    """Load trained model or train new one"""
    try:
        # Try to load existing model
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('iris_scaler.pkl')
        return model, scaler
    
    except:
        # Train new model if not found
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = SVC(kernel='rbf', random_state=42, probability=True)
        model.fit(X_train_scaled, y_train)

        # Save model
        joblib.dump(model, 'iris_model.pkl')
        joblib.dump(scaler, 'iris_scaler.pkl')
        
        return model, scaler

def main():
    # Main header
    st.markdown('<h1 class="main-header">Iris Flower Classification</h1>', unsafe_allow_html=True)

    # Load data and model
    df = load_data()
    model, scaler = load_or_train_model()

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "📈 Data Analysis", "🔮 Prediction", "📊 Model Performance"]
    )

    if page == "🏠 Home":
        show_homepage(df)
    elif page == "📈 Data Analysis":
        show_data_analysis(df)
    elif page == "🔮 Prediction":
        show_prediction_page(model, scaler)
    elif page == "📊 Model Performance":
        show_model_performance()

def show_homepage(df):
    """Home page"""
    st.markdown("## 🎯 About This Project")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This is a **practice project** for learning data science concepts through the classic Iris flower classification problem.
        
        ### 🌸 Iris Dataset
        - **150 samples** (50 from each class)
        - **4 features**: Sepal length, Sepal width, Petal length, Petal width
        - **3 classes**: Setosa, Versicolor, Virginica)

        ### ✨ Features
        - ✅ Data exploration and visualization
        - ✅ Machine learning model development
        - ✅ Interactive web application
        - ✅ Real-time predictions
        """)

    with col2:
        st.markdown('### 📝 Dataset Summary')
        st.metric("Total Samples", len(df))
        st.metric("Features", 4)
        st.metric("Classes", 3)

        st.markdown("### 🎯 Class Distribution")
        species_counts = df['species'].value_counts()
        fig = px.pie(values=species_counts.values, names = species_counts.index,
                    title="Class Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_data_analysis(df):
    """Data analysis page"""
    st.markdown("## 📈 Data Analysis")

    # Basic statistics
    st.markdown("### 📊 Basic Statistics")
    st.dataframe(df.describe())

    # Class distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Class Distribution")
        species_counts = df['species'].value_counts()
        fig = px.bar(x=species_counts.index, y=species_counts.values,
        title="Iris Species Count")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📈 Feature Distribution")
        feature = st.selectbox("Select Feature", df.columns[:-2])
        fig = px.histogram(df, x=feature, color='species',
                          title=f"{feature} Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.markdown('### 🔍 Feature Relationships')
    col1, col2 = st.columns(2)

    with col1:
        x_feature = st.selectbox("X Feature", df.columns[:-2], index=0)
    with col2:
        y_feature = st.selectbox("Y feature", df.columns[:-2], index=1)

    fig = px.scatter(df, x=x_feature, y=y_feature, color='species',
                    title=f"{x_feature} vs {y_feature}")
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model, scaler):
    """Prediction page"""
    st.markdown("## 🔮 Iris Species Prediction")

    if model is None or scaler is None:
        st.error("Model not loaded: Please run model_development.py first")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📝 Enter Flower Measurements")

        # Initialize session state for example values
        if 'sepal_length' not in st.session_state:
            st.session_state.sepal_length = 5.4
        if 'sepal_width' not in st.session_state:
            st.session_state.sepal_width = 3.4
        if 'petal_length' not in st.session_state:
            st.session_state.petal_length = 4.7
        if 'petal_width' not in st.session_state:
            st.session_state.petal_width = 1.4
        
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, st.session_state.sepal_length, 0.1, key="sepal_length_slider")
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, st.session_state.sepal_width, 0.1, key="sepal_width_slider")
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, st.session_state.petal_length, 0.1, key="petal_length_slider")
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, st.session_state.petal_width, 0.1, key="petal_width_slider")

        if st.button("🔮 Predict Species", type="primary"):
            # Make prediction
            sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            sample_scaled = scaler.transform(sample)
            prediction = model.predict(sample_scaled)
            probability = model.predict_proba(sample_scaled) if hasattr(model, 'predict_proba') else None

            species_names = ['setosa', 'versicolor', 'virginica']
            predicted_species = species_names[prediction[0]]

            # Show results
            with col2:
                st.markdown("### 🎯 Prediction Result")

                # Prediction box
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🌺 Predicted Species: <strong>{predicted_species.title()}</strong></h3>
                </div>
                """, unsafe_allow_html=True)

                # Probability chart
                if probability is not None:
                    fig = px.bar(x=species_names, y=probability[0],
                                title="Class Probabilities",
                                color=probability[0],
                                color_continuous_scale='viridis')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                # Probability table
                st.markdown("### 📊 Detailed Probabilitiees")
                if probability is not None:
                    prob_df = pd.DataFrame({
                        'Species': species_names,
                        'Probability': probability[0]
                    })
                    st.dataframe(prob_df, use_container_width=True)
        
    # Example values
    st.markdown("### 💡 Example Values")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Setosa Example"):
            st.session_state.sepal_length = 5.1
            st.session_state.sepal_width = 3.5
            st.session_state.petal_length = 1.4
            st.session_state.petal_width = 0.2
            st.rerun()

    with col2:
        if st.button("Versicolor Example"):
            st.session_state.sepal_length = 6.0
            st.session_state.sepal_width = 2.2
            st.session_state.petal_length = 4.0
            st.session_state.petal_width = 1.0
            st.rerun()

    with col3:
        if st.button("Virginica Example"):
            st.session_state.sepal_length = 6.3
            st.session_state.sepal_width = 3.3
            st.session_state.petal_length = 6.0
            st.session_state.petal_width = 2.5
            st.rerun()

def show_model_performance():
    """Model performance page"""
    st.markdown("## 📊 Model Performance")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>96.67%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 F1-Score</h3>
            <h1>0.967</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <h2>~1ms</h2>
        </div>
        """, unsafe_allow_html=True)

    # Model details
    st.markdown("### ℹ️ Model Information")
    st.markdown("""
    **Support Vector Machine (SVM)**
    - Kernel: RBF (Radial Basis Function)
    - Cross-validation score: 96.67%
    - Best performing model among tested algorithms
    """)

    # Performance by class
    st.markdown("### 📊 Performance by Class")
    class_performance = {
        'Class': ['Setosa', 'Versicolor', 'Virginica'],
        'Precision': [1.00, 1.00, 0.91],
        'Recall': [1.00, 0.90, 1.00],
        'F1-Score': [1.00, 0.95, 0.95]    
        }

    perf_df = pd.DataFrame(class_performance)
    st.dataframe(perf_df, use_container_width=True)

if __name__ == "__main__":
    main()