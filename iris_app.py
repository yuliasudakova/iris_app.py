import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Iris Dataset Explorer", page_icon="🌸", layout="wide")

# Load dataset
df = pd.read_csv("iris.csv")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page",
    [
        "Home",
        "Data Overview",
        "Exploratory Data Analysis",
        "Model Training and Evaluation",
        "Make Predictions!",
        "Extras"
    ]
)

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.title("📊 Iris Dataset Explorer")

    st.subheader("Welcome to our Iris dataset explorer app!")

    st.write(
        """
        This app provides an interactive platform to explore the famous Iris dataset.
        You can visualize the distribution of data and explore relationships between features.
        Use the sidebar to navigate through the sections.
        """
    )

    st.image(
        "https://bouqs.com/blog/wp-content/uploads/2021/11/iris-flower-meaning-and-symbolism.jpg",
        caption="The Iris Flower"
    )

# -------------------------
# Data Overview
# -------------------------
elif page == "Data Overview":

    st.title("🔢 Data Overview")

    st.subheader("About the Data")

    st.write(
        """
        The Iris dataset is a classic dataset in machine learning.
        It contains 150 samples from three iris species:
        setosa, versicolor, and virginica.
        """
    )

    if st.checkbox("Show DataFrame"):
        st.dataframe(df)

    if st.checkbox("Show Shape of Data"):
        st.write(df.shape)

# -------------------------
# Exploratory Data Analysis
# -------------------------
elif page == "Exploratory Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    x = st.selectbox("X Axis", num_cols)
    y = st.selectbox("Y Axis", num_cols)

    fig = px.scatter(df, x=x, y=y, color="species")

    st.plotly_chart(fig)

# -------------------------
# Model Training
# -------------------------
elif page == "Model Training and Evaluation":

    st.title("🛠️ Model Training and Evaluation")

    model_option = st.sidebar.selectbox(
        "Select Model",
        ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"]
    )

    X = df.drop(columns="species")
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_option == "K-Nearest Neighbors":

        k = st.sidebar.slider("K", 1, 20, 5)

        model = KNeighborsClassifier(n_neighbors=k)

    elif model_option == "Logistic Regression":

        model = LogisticRegression()

    else:

        model = RandomForestClassifier()

    model.fit(X_train_scaled, y_train)

    st.write("Training Accuracy:", model.score(X_train_scaled, y_train))

    st.write("Test Accuracy:", model.score(X_test_scaled, y_test))

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test_scaled,
        y_test,
        ax=ax
    )

    st.pyplot(fig)

# -------------------------
# Make Predictions
# -------------------------
elif page == "Make Predictions!":

    st.title("🌸 Make Predictions")

    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)

    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)

    petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)

    petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

    user_input = pd.DataFrame(
        {
            "sepal_length": [sepal_length],
            "sepal_width": [sepal_width],
            "petal_length": [petal_length],
            "petal_width": [petal_width],
        }
    )

    st.write("Your Input")

    st.dataframe(user_input)

    X = df.drop(columns="species")
    y = df["species"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    user_scaled = scaler.transform(user_input)

    model = KNeighborsClassifier(n_neighbors=9)

    model.fit(X_scaled, y)

    prediction = model.predict(user_scaled)[0]

    st.success(f"Prediction: {prediction}")

    st.balloons()

# -------------------------
# Extras
# -------------------------
elif page == "Extras":

    st.title("Extras")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://static.streamlit.io/examples/cat.jpg")

    with col2:
        st.image("https://static.streamlit.io/examples/dog.jpg")

    with col3:
        st.image("https://static.streamlit.io/examples/owl.jpg")

    tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

    with tab1:
        st.write("First tab")

    with tab2:
        st.write("Second tab")