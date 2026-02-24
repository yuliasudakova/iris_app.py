import streamlit as st
import pandas as pd
import plotly.express as px

# ✅ MUST be the first Streamlit command and only called once
st.set_page_config(page_title="Iris Dataset Explorer", page_icon="🌸", layout="wide")

# ✅ Load dataset (your repo shows iris.csv is in the ROOT)
df = pd.read_csv("iris.csv")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a Page",
    ["Home", "Data Overview", "Exploratory Data Analysis", "Extras"]
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
        caption="The Iris Flower",
        use_container_width=True
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
        The Iris dataset is one of the most famous datasets in the literature of machine learning and data analysis. 
        It contains 150 samples of iris flowers from three different species (Iris-setosa, Iris-versicolor, Iris-virginica). 
        For each flower, the dataset includes the length and width of the sepals and petals.
        """
    )
    st.image(
        "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png",
        caption="Iris Dataset",
        use_container_width=True
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

    # Handle if species column name is not exactly 'species'
    # (But your charts assume 'species', so we check it)
    if "species" not in df.columns:
        st.warning("Column 'species' was not found in your CSV. Some charts may not work.")
        # If you want, you could rename automatically if you find a similar column.

    if "Histograms" in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numerical column for the histogram:", num_cols, key="hist_col")
        chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"

        if "species" in df.columns and st.checkbox("Show by Species"):
            st.plotly_chart(
                px.histogram(df, x=h_selected_col, color="species", title=chart_title, barmode="overlay"),
                use_container_width=True
            )
        else:
            st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title), use_container_width=True)

    if "Box Plots" in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numerical column for the box plot:", num_cols, key="box_col")
        chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"

        if "species" in df.columns:
            st.plotly_chart(
                px.box(df, x="species", y=b_selected_col, title=chart_title, color="species"),
                use_container_width=True
            )
        else:
            st.plotly_chart(px.box(df, y=b_selected_col, title=chart_title), use_container_width=True)

    if "Scatterplots" in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, key="scatter_x")
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, key="scatter_y")
        chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"

        if "species" in df.columns:
            st.plotly_chart(
                px.scatter(df, x=selected_col_x, y=selected_col_y, color="species", title=chart_title),
                use_container_width=True
            )
        else:
            st.plotly_chart(
                px.scatter(df, x=selected_col_x, y=selected_col_y, title=chart_title),
                use_container_width=True
            )

    if "Count Plots" in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        if len(obj_cols) == 0:
            st.info("No categorical (object) columns found to plot.")
        else:
            selected_col = st.selectbox("Select a categorical variable:", obj_cols, key="count_col")
            chart_title = f"Distribution of {selected_col.title()}"

            if "species" in df.columns and selected_col != "species":
                st.plotly_chart(
                    px.histogram(df, x=selected_col, color="species", title=chart_title),
                    use_container_width=True
                )
            else:
                st.plotly_chart(px.histogram(df, x=selected_col, title=chart_title), use_container_width=True)

# -------------------------
# Extras
# -------------------------
elif page == "Extras":
    st.title("Adding Columns")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg", use_container_width=True)

    with col2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg", use_container_width=True)

    with col3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg", use_container_width=True)

    st.divider()

    st.title("Adding Tabs")

    tab1, tab2, tab3 = st.tabs(["First Tab", "Second Tab", "Third Tab"])

    with tab1:
        st.write("Place whatever you want here! This is the first tab!")

    with tab2:
        st.write("This is tab 2!")
        st.image("https://static.streamlit.io/examples/owl.jpg", use_container_width=True)

    with tab3:
        st.write("The best for last")
        st.balloons()

    st.divider()

    st.title("Adding a Container")

    container = st.container(border=True)
    container.write("This is inside the container")
    st.write("This is outside the container")
    container.write("This is inside too")
