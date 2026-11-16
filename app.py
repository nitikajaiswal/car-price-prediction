

import io
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AutoValue | Car Price Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PREMIUM AUTOMOTIVE THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');

    :root {
        --black: #111110;
        --dark: #1c1b19;
        --gold: #c5a46d;
        --champagne: #e8d7b7;
        --cream: #f4eee4;
    }
    .stApp {
        background: linear-gradient(135deg, #111110, #201d19 65%, #30261e);
        color: var(--cream);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #171614;
        border-right: 1px solid #3c3328;
    }
    [data-testid="stSidebar"] * { color: #e8d7b7; }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #e8d7b7 !important;
    }
    p, label, span { color: #e8e0d3; }
    .hero {
        padding: 42px;
        border-radius: 22px;
        background: linear-gradient(100deg, rgba(15,15,14,.97), rgba(50,39,28,.87));
        border: 1px solid #7f6745;
        margin-bottom: 26px;
        box-shadow: 0 15px 45px rgba(0,0,0,.3);
    }
    .hero h1 {
        font-size: 48px !important;
        margin-bottom: 5px;
        color: #e8d7b7 !important;
    }
    .hero p { color: #c9b99e; font-size: 16px; }
    .eyebrow {
        color: #c5a46d;
        letter-spacing: 4px;
        font-size: 12px;
        font-weight: 700;
    }
    .panel {
        background: #211f1b;
        border: 1px solid #514332;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 15px;
    }
    .stat {
        background: linear-gradient(145deg, #28241e, #171614);
        border: 1px solid #675239;
        padding: 20px;
        border-radius: 16px;
        min-height: 125px;
    }
    .stat-label { color: #b5a58e; font-size: 12px; letter-spacing: 1px; }
    .stat-value { font-size: 25px; font-weight: 700; color: #e8d7b7; margin-top: 8px; }
    div.stButton > button, div.stDownloadButton > button, button[kind="primary"] {
        background: linear-gradient(100deg, #b99861, #dfc89e);
        color: #171614 !important;
        font-weight: 700;
        border: none;
        border-radius: 9px;
        min-height: 44px;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #f0dfbd;
        color: #111 !important;
    }
    div[data-testid="stMetric"] {
        background: #211f1b;
        border: 1px solid #514332;
        border-radius: 12px;
        padding: 15px;
    }
    div[data-testid="stMetric"] label { color: #c5a46d !important; }
    div[data-testid="stMetricValue"] { color: #e8d7b7 !important; }
    input, textarea { background: #25221d !important; color: #f4eee4 !important; }
    [data-baseweb="select"] > div { background-color: #25221d; border-color: #766044; }
    hr { border-color: #504332; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def clean_number(value):
    """Convert values such as '1,20,000 km' to a number."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def load_dataset(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data.columns = [str(col).strip() for col in data.columns]
    data = data.loc[:, ~pd.Index(data.columns).str.lower().str.startswith("unnamed")]

    for column in data.columns:
        if data[column].dtype == "object":
            converted = data[column].map(clean_number)
            non_empty = data[column].notna().sum()
            if non_empty and converted.notna().sum() / non_empty >= 0.80:
                data[column] = converted
            else:
                data[column] = data[column].astype("string").str.strip()
    return data


def find_target(columns):
    candidates = ["selling_price", "price", "car_price", "sellingprice"]
    lower_map = {str(column).lower().replace(" ", "_"): column for column in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    for column in columns:
        if "price" in str(column).lower():
            return column
    return columns[-1]


def engineer_features(data):
    """Apply the same feature engineering to training and prediction data."""
    data = data.copy()

    name_column = next(
        (c for c in data.columns if str(c).lower() in ["name", "car_name", "model"]),
        None,
    )
    if name_column:
        data["brand"] = (
            data[name_column].astype("string").str.strip().str.split().str[0].str.lower()
        )
        data.drop(columns=[name_column], inplace=True)

    year_column = next(
        (c for c in data.columns if str(c).lower() in ["year", "model_year"]),
        None,
    )
    if year_column:
        year_values = pd.to_numeric(data[year_column].map(clean_number), errors="coerce")
        data["car_age"] = (pd.Timestamp.now().year - year_values).clip(lower=0, upper=80)
        data.drop(columns=[year_column], inplace=True)

    for column in data.columns:
        column_name = str(column).lower()
        if any(
            word in column_name
            for word in ["mileage", "engine", "max_power", "horsepower", "km_driven", "power", "torque"]
        ):
            data[column] = data[column].map(clean_number)

    for column in list(data.columns):
        if str(column).lower() in ["id", "car_id", "listing_id"]:
            data.drop(columns=[column], inplace=True)

    return data


def build_model(features, algorithm):
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [c for c in features.columns if c not in numeric_columns]

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessing = ColumnTransformer(transformers=transformers)

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    return Pipeline(
        steps=[("preprocessing", preprocessing), ("regression", models[algorithm])]
    )


def format_price(value):
    return f"₹{value:,.0f}"


# =========================================================
# SESSION STATE
# =========================================================
defaults = {
    "dataset": None,
    "model": None,
    "raw_columns": None,
    "target": None,
    "metrics": None,
    "algorithm": None,
    "prediction": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <h2 style="font-family:Playfair Display;color:#d8bd8f;">AUTOVALUE</h2>
        <p style="letter-spacing:3px;font-size:10px;color:#a68b60;">
        CAR PRICE INTELLIGENCE</p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Home", "Dataset Explorer", "Model Training", "Price Prediction", "About Project"],
    )
    st.divider()
    st.markdown("### SYSTEM STATUS")
    if st.session_state.model is not None:
        st.success("Model trained")
    else:
        st.info("Waiting for training")
    st.caption("AUTOVALUE • ML EDITION")

# =========================================================
# HOME PAGE
# =========================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">INTELLIGENT AUTOMOTIVE ANALYTICS</div>
            <h1>Know Your Car's<br>True Worth.</h1>
            <p>A machine-learning powered car valuation system designed to estimate
            vehicle prices through intelligent data analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## Welcome to AutoValue")
    st.write(
        "Discover the estimated market value of a vehicle using its specifications, "
        "history and performance."
    )

    col1, col2, col3 = st.columns(3)
    for column, number, heading, description in [
        (col1, "01 / DATA", "Explore", "Analyze automotive datasets and discover patterns."),
        (col2, "02 / INTELLIGENCE", "Train", "Build regression models using machine learning."),
        (col3, "03 / VALUATION", "Predict", "Estimate vehicle prices using trained models."),
    ]:
        with column:
            st.markdown(
                f'<div class="stat"><div class="stat-label">{number}</div>'
                f'<div class="stat-value">{heading}</div><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("## What influences a car's price?")
    factors = st.columns(4)
    for column, heading, description in zip(
        factors,
        ["BRAND", "ENGINE", "MILEAGE", "AGE"],
        [
            "Brand reputation and goodwill",
            "Engine capacity and horsepower",
            "Distance travelled and fuel efficiency",
            "Manufacturing year and ownership",
        ],
    ):
        with column:
            st.markdown(
                f'<div class="panel"><h3>{heading}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )
    st.info("Start by uploading your car dataset in the Dataset Explorer.")

# =========================================================
# DATASET EXPLORER
# =========================================================
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.write("Upload your automotive CSV dataset and explore its structure.")

    uploaded_file = st.file_uploader("Upload car dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            new_data = load_dataset(uploaded_file)
            st.session_state.dataset = new_data
            st.session_state.model = None
            st.session_state.metrics = None
            st.session_state.raw_columns = None
            st.success("Dataset uploaded successfully.")
        except Exception as error:
            st.error(f"Dataset error: {error}")

    data = st.session_state.dataset
    if data is not None:
        st.markdown("### Dataset Overview")
        a, b, c = st.columns(3)
        a.metric("Total Records", f"{len(data):,}")
        b.metric("Total Features", len(data.columns))
        c.metric("Missing Values", int(data.isnull().sum().sum()))

        st.markdown("### Dataset Preview")
        st.markdown(
            data.head(15).to_html(index=False, border=0, classes="car-data-table"),
            unsafe_allow_html=True,
        )

        st.markdown("### Missing Value Analysis")
        missing = data.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if len(missing):
            fig = px.bar(
                x=missing.index,
                y=missing.values,
                labels={"x": "Features", "y": "Missing values"},
                color_discrete_sequence=["#c5a46d"],
            )
            fig.update_layout(
                paper_bgcolor="#211f1b",
                plot_bgcolor="#211f1b",
                font_color="#e8d7b7",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found.")

        numeric_columns = data.select_dtypes(include="number").columns.tolist()
        if numeric_columns:
            st.markdown("### Feature Distribution")
            selected_column = st.selectbox("Select a feature", numeric_columns)
            fig = px.histogram(
                data,
                x=selected_column,
                color_discrete_sequence=["#c5a46d"],
                title=f"Distribution of {selected_column}",
            )
            fig.update_layout(
                paper_bgcolor="#211f1b",
                plot_bgcolor="#211f1b",
                font_color="#e8d7b7",
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODEL TRAINING
# =========================================================
elif page == "Model Training":
    st.title("Machine Learning Studio")
    st.write(
        "Train a regression model to learn the relationship between car "
        "specifications and selling prices."
    )

    data = st.session_state.dataset
    if data is None:
        st.warning("Please upload your dataset in Dataset Explorer first.")
    elif len(data) < 10:
        st.error("Please upload a dataset containing at least 10 records.")
    else:
        guessed_target = find_target(data.columns)
        target = st.selectbox(
            "Select target price column",
            options=list(data.columns),
            index=list(data.columns).index(guessed_target),
        )
        algorithm = st.selectbox(
            "Choose regression algorithm",
            ["Random Forest", "Extra Trees", "Gradient Boosting"],
        )
        test_size = st.slider("Test dataset percentage", 10, 40, 20, 5)

        st.markdown("### Training Configuration")
        st.write(
            "The pipeline automatically handles missing values, categorical "
            "encoding and numerical preprocessing."
        )
        st.info("Your dataset is ready for machine learning.")

        if st.button(
            "TRAIN MACHINE LEARNING MODEL",
            key="actual_training_button",
            use_container_width=True,
        ):
            try:
                working = data.copy()
                working[target] = working[target].map(clean_number)
                working = working.dropna(subset=[target])

                if len(working) < 10:
                    st.error("At least 10 valid records are recommended.")
                else:
                    y = working[target].astype(float)
                    raw_features = working.drop(columns=[target])
                    features = engineer_features(raw_features)

                    # Remove unusable columns before building the pipeline.
                    features = features.dropna(axis=1, how="all")
                    features = features.loc[
                        :, features.nunique(dropna=False) > 1
                    ]

                    if features.shape[1] == 0:
                        st.error("No usable feature columns were found.")
                    else:
                        actual_test_size = max(1, int(round(len(working) * test_size / 100)))
                        if actual_test_size >= len(working):
                            actual_test_size = len(working) - 1

                        X_train, X_test, y_train, y_test = train_test_split(
                            features,
                            y,
                            test_size=actual_test_size,
                            random_state=42,
                        )

                        model_pipeline = build_model(X_train, algorithm)
                        with st.spinner("Training model. Please wait..."):
                            model_pipeline.fit(X_train, y_train)
                            predictions = model_pipeline.predict(X_test)

                        mae = mean_absolute_error(y_test, predictions)
                        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
                        r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0.0

                        st.session_state.model = model_pipeline
                        st.session_state.raw_columns = raw_features.columns.tolist()
                        st.session_state.target = target
                        st.session_state.algorithm = algorithm
                        st.session_state.metrics = {
                            "mae": float(mae),
                            "rmse": float(rmse),
                            "r2": float(r2),
                            "actual": y_test.tolist(),
                            "predicted": predictions.tolist(),
                        }
                        st.success("Model trained successfully!")
            except Exception as error:
                st.error(f"Training error: {error}")

        # Display evaluation results after a successful training run.
        if st.session_state.model is not None and st.session_state.metrics is not None:
            metrics = st.session_state.metrics
            st.markdown("---")
            st.markdown("## Model Performance")
            a, b, c = st.columns(3)
            a.metric("MAE", format_price(metrics["mae"]))
            b.metric("RMSE", format_price(metrics["rmse"]))
            c.metric("R² Score", f'{metrics["r2"]:.3f}')
            st.caption(f'Algorithm: {st.session_state.algorithm}')

            st.markdown("### Actual vs Predicted Prices")
            chart_data = pd.DataFrame(
                {
                    "Actual Price": metrics["actual"],
                    "Predicted Price": metrics["predicted"],
                }
            )
            fig = px.scatter(
                chart_data,
                x="Actual Price",
                y="Predicted Price",
                opacity=0.7,
                color_discrete_sequence=["#d0ad72"],
                title="Regression Model Performance",
            )
            low = min(chart_data.min().min(), chart_data.min().min())
            high = max(chart_data.max().max(), chart_data.max().max())
            fig.add_shape(
                type="line",
                x0=low,
                y0=low,
                x1=high,
                y1=high,
                line=dict(color="#f0dfbd", dash="dash"),
            )
            fig.update_layout(
                paper_bgcolor="#211f1b",
                plot_bgcolor="#211f1b",
                font_color="#e8d7b7",
            )
            st.plotly_chart(fig, use_container_width=True)

            buffer = io.BytesIO()
            joblib.dump(st.session_state.model, buffer)
            st.download_button(
                "DOWNLOAD MODEL",
                data=buffer.getvalue(),
                file_name="autovalue_model.joblib",
                mime="application/octet-stream",
            )

# =========================================================
# PRICE PREDICTION
# =========================================================
elif page == "Price Prediction":
    st.title("Vehicle Valuation")
    st.write("Enter vehicle specifications to estimate its selling price.")

    model = st.session_state.model
    data = st.session_state.dataset

    if model is None or data is None:
        st.warning("Upload a dataset and train a model first.")
    else:
        raw_columns = st.session_state.raw_columns
        raw_data = data.drop(
            columns=[st.session_state.target],
            errors="ignore",
        )

        input_values = {}
        with st.form("prediction_form"):
            left, right = st.columns(2)
            for index, column in enumerate(raw_columns):
                if column not in raw_data.columns:
                    continue
                series = raw_data[column]
                current_column = left if index % 2 == 0 else right
                label = str(column).replace("_", " ").title()

                with current_column:
                    if pd.api.types.is_numeric_dtype(series):
                        median = series.median()
                        default_value = float(median) if pd.notna(median) else 0.0
                        input_values[column] = st.number_input(
                            label,
                            value=default_value,
                        )
                    else:
                        options = (
                            series.dropna().astype(str).unique().tolist()
                        )
                        options = sorted(options)
                        if options and len(options) <= 35:
                            input_values[column] = st.selectbox(label, options)
                        else:
                            input_values[column] = st.text_input(
                                label,
                                placeholder=f"Enter {label.lower()}",
                            )

            submitted = st.form_submit_button("ESTIMATE CAR PRICE")

        if submitted:
            try:
                input_frame = pd.DataFrame([input_values])
                prepared = engineer_features(input_frame)

                # Align prediction columns with the columns used during training.
                trained_features = model.named_steps["preprocessing"].feature_names_in_
                for column in trained_features:
                    if column not in prepared.columns:
                        prepared[column] = np.nan
                prepared = prepared[list(trained_features)]

                prediction = float(model.predict(prepared)[0])
                st.session_state.prediction = prediction

                st.markdown("---")
                st.markdown(
                    f"""
                    <div class="panel" style="text-align:center;padding:40px;">
                        <p style="color:#c5a46d;letter-spacing:3px;">
                            ESTIMATED SELLING PRICE
                        </p>
                        <h1 style="font-size:52px;">{format_price(prediction)}</h1>
                        <p>Generated using your trained {st.session_state.algorithm} model.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    "This is a statistical estimate, not a guaranteed market price. "
                    "Actual value also depends on vehicle condition, location, "
                    "documentation and demand."
                )
            except Exception as error:
                st.error(f"Prediction error: {error}")

# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "About Project":
    st.title("About AutoValue")
    st.markdown(
        """
        ## Project Overview
        AutoValue is a machine-learning application designed to estimate used-car
        selling prices using vehicle specifications and historical data.

        ## Project Objectives
        - Understand regression-based machine learning.
        - Perform data preprocessing and handle missing values.
        - Perform feature engineering.
        - Train regression algorithms.
        - Evaluate model performance.
        - Generate predictions for new vehicles.

        ## Machine Learning Algorithms
        **1. Random Forest Regression**  
        Combines predictions from multiple decision trees.

        **2. Extra Trees Regression**  
        Uses randomized decision trees to learn complex relationships.

        **3. Gradient Boosting Regression**  
        Builds sequential models to reduce prediction errors.

        ## Technologies Used
        Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly and Joblib.

        ## Model Evaluation
        **MAE:** Average absolute difference between actual and predicted prices.  
        **RMSE:** Measures prediction errors while penalizing larger errors.  
        **R²:** Measures how much variation in the target is explained by the model.
        """
    )
    st.info(
        "AutoValue demonstrates how machine learning can support data-driven "
        "automotive price estimation."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <p style="text-align:center;color:#9d8c73;letter-spacing:2px;font-size:11px;">
    AUTOVALUE · CAR PRICE INTELLIGENCE<br>
    POWERED BY PYTHON AND MACHINE LEARNING
    </p>
    """,
    unsafe_allow_html=True,
)
