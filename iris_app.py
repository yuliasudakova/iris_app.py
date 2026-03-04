import streamlit as st
import pandas as pd
import plotly.express as px

# ✅ Must be the first Streamlit command (and only called once)
st.set_page_config(page_title="Iris Dataset Explorer", page_icon="🌸", layout="wide")

# ✅ Load dataset (your repo has iris.csv in the ROOT)
df = pd.read_csv("iris.csv")


# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page", 
    ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!", "Extras"]
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
        You can visualize the distribution of data, explore relationships between features.
        Use the sidebar to navigate through the sections.
        """
    )
    st.image(
        "https://bouqs.com/blog/wp-content/uploads/2021/11/iris-flower-meaning-and-symbolism.jpg",
        caption="The Iris Flower"
    )
    st.write("Use the sidebar to navigate between different sections.")

# -------------------------
# Data Overview
# -------------------------
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write(
        """
        The Iris dataset is a classic dataset in machine learning and data analysis.
        It contains 150 samples from three iris species (setosa, versicolor, virginica).
        Each sample includes sepal length/width and petal length/width.
        """
    )
    st.image(
        "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png",
        caption="Iris Dataset"
    )

    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df, use_container_width=True)

    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# -------------------------
# Exploratory Data Analysis (EDA)
# -------------------------
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect(
        "Visualization Options",
        ["Histograms", "Box Plots", "Scatterplots", "Count Plots"]
    )

    obj_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Ensure we have a species column for color grouping
    if "species" not in df.columns:
        st.warning("Column 'species' not found. Species-based plots will be limited.")

    if "Histograms" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_col = st.selectbox("Select a numerical column:", num_cols, key="hist_col")
        title = f"Distribution of {h_col.replace('_', ' ').title()}"

        if "species" in df.columns and st.checkbox("Show by Species"):
            fig = px.histogram(df, x=h_col, color="species", barmode="overlay", title=title)
        else:
            fig = px.histogram(df, x=h_col, title=title)
        st.plotly_chart(fig, use_container_width=True)

    if "Box Plots" in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_col = st.selectbox("Select a numerical column:", num_cols, key="box_col")
        title = f"Distribution of {b_col.replace('_', ' ').title()}"

        if "species" in df.columns:
            fig = px.box(df, x="species", y=b_col, color="species", title=title)
        else:
            fig = px.box(df, y=b_col, title=title)
        st.plotly_chart(fig, use_container_width=True)

    if "Scatterplots" in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        x_col = st.selectbox("Select x-axis variable:", num_cols, key="scatter_x")
        y_col = st.selectbox("Select y-axis variable:", num_cols, key="scatter_y")
        title = f"{x_col.replace('_', ' ').title()} vs. {y_col.replace('_', ' ').title()}"

        if "species" in df.columns:
            fig = px.scatter(df, x=x_col, y=y_col, color="species", title=title)
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        st.plotly_chart(fig, use_container_width=True)

    if "Count Plots" in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        if not obj_cols:
            st.info("No categorical columns found.")
        else:
            c_col = st.selectbox("Select a categorical column:", obj_cols, key="count_col")
            title = f"Distribution of {c_col.replace('_', ' ').title()}"

            # If species exists and selected column isn't species, color by species
            if "species" in df.columns and c_col != "species":
                fig = px.histogram(df, x=c_col, color="species", title=title)
            else:
                fig = px.histogram(df, x=c_col, title=title)
            st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Extras
# -------------------------
elif page == "Extras":
    st.title("Extras: Layout & Fun Widgets")

    st.subheader("Adding Columns")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg")

    with col2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg")

    with col3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg")

    st.divider()

    st.subheader("Adding Tabs")
    tab1, tab2, tab3 = st.tabs(["First Tab", "Second Tab", "Third Tab"])

    with tab1:
        st.write("Place whatever you want here! This is the first tab!")

    with tab2:
        st.write("This is tab 2!")
        st.image("https://static.streamlit.io/examples/owl.jpg")

    with tab3:
        st.write("The best for last")
        st.balloons()

    st.divider()

    st.subheader("Adding a Container")
    container = st.container(border=True)
    container.write("This is inside the container")
    st.write("This is outside the container")
    container.write("This is inside too")
