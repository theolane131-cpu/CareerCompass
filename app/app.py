"""
Career Compass Streamlit Application

This application loads a trained Random Forest regression model and uses
employee and company information entered by the user to estimate an annual
salary.

Team DataForge
University of Oklahoma
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
# This command must appear before other Streamlit commands.

st.set_page_config(
    page_title="Career Compass",
    page_icon="🧭",
    layout="wide",
)


# ---------------------------------------------------------
# OU-inspired color theme
# ---------------------------------------------------------
# The interface uses OU crimson with a light ivory page background.
# White is retained for cards and form elements so they stand out.

OU_CRIMSON = "#841617"
OU_DARK_CRIMSON = "#4E0002"
OU_DARK_GRAY = "#323232"
OU_LIGHT_GRAY = "#F0F0F0"
OU_BACKGROUND = "#FAF8F4"
OU_WHITE = "#FFFFFF"


# ---------------------------------------------------------
# Custom interface styling
# ---------------------------------------------------------
# Streamlit handles the app functionality, while the CSS below provides
# the OU-inspired colors, spacing, buttons, and prediction card.

st.markdown(
    f"""
    <style>
        /* Main application page */
        .stApp {{
            background-color: {OU_BACKGROUND};
            color: {OU_DARK_GRAY};
        }}

        /* Restrict the maximum page width for better readability */
        .block-container {{
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        /* Main application header */
        .career-header {{
            background: linear-gradient(
                135deg,
                {OU_CRIMSON},
                {OU_DARK_CRIMSON}
            );
            padding: 2rem;
            border-radius: 14px;
            color: {OU_WHITE};
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.16);
        }}

        .career-header h1 {{
            color: {OU_WHITE};
            margin: 0;
            font-size: 2.5rem;
        }}

        .career-header h3 {{
            color: {OU_WHITE};
            margin-top: 0.35rem;
            font-weight: 400;
        }}

        .career-header p {{
            color: {OU_WHITE};
            margin-bottom: 0;
            font-size: 1.05rem;
        }}

        /* Section headings */
        h2, h3 {{
            color: {OU_CRIMSON};
        }}

        /* Input labels */
        label {{
            font-weight: 600 !important;
            color: {OU_DARK_GRAY} !important;
        }}

        /* Dropdown input containers */
        div[data-baseweb="select"] > div {{
            background-color: {OU_WHITE};
            border-radius: 8px;
        }}

        /* Main prediction button */
        div.stButton > button {{
            background-color: {OU_CRIMSON};
            color: {OU_WHITE};
            border: none;
            border-radius: 8px;
            padding: 0.7rem 1.6rem;
            font-weight: 700;
            font-size: 1rem;
        }}

        div.stButton > button:hover {{
            background-color: {OU_DARK_CRIMSON};
            color: {OU_WHITE};
            border: none;
        }}

        /* Prediction result card */
        .prediction-card {{
            background-color: {OU_LIGHT_GRAY};
            border-left: 8px solid {OU_CRIMSON};
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        }}

        .prediction-label {{
            color: {OU_DARK_GRAY};
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }}

        .prediction-value {{
            color: {OU_CRIMSON};
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0;
        }}

        /* Informational banner near the top */
        .info-box {{
            background-color: {OU_WHITE};
            border-left: 5px solid {OU_CRIMSON};
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        }}

        /* Hide selected default Streamlit branding */
        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
# pathlib creates reliable file paths across Windows, macOS, Linux,
# and potential future deployment environments.

APP_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIRECTORY.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_salary_model.joblib"
)


# ---------------------------------------------------------
# Load the saved model bundle
# ---------------------------------------------------------
# The Joblib bundle contains both:
# 1. The trained Random Forest model
# 2. The feature columns expected by the model
#
# The cache prevents Streamlit from reloading the model every time the
# user changes an input.

@st.cache_resource
def load_model_bundle():
    """
    Load the trained Random Forest model and its expected feature columns.

    Returns
    -------
    dict
        A dictionary containing the trained model and feature-column list.

    Raises
    ------
    FileNotFoundError
        If the model bundle cannot be found in the models directory.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model_bundle = load_model_bundle()

    salary_model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

except (FileNotFoundError, KeyError, OSError) as error:
    st.error(
        "The salary prediction model could not be loaded. "
        "Confirm that the model file exists in the models folder."
    )
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Application header
# ---------------------------------------------------------

st.markdown(
    """
    <div class="career-header">
        <h1>🧭 Career Compass</h1>
        <h3>Employee Salary Prediction System</h3>
        <p>
            Estimate employee salaries using a trained Random Forest
            machine learning model built from historical data science
            salary information.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Explain how the prediction should be interpreted.
st.markdown(
    """
    <div class="info-box">
        This application provides a decision-support estimate rather
        than an exact compensation recommendation. Actual salaries may
        also depend on education, certifications, negotiation, benefits,
        industry, company policies, and local market conditions.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Display values used by the dropdown menus
# ---------------------------------------------------------
# The model expects abbreviated values such as SE, FT, and US.
# format_func displays readable descriptions while preserving the
# original codes required by the model.

EXPERIENCE_LABELS = {
    "EN": "Entry-Level",
    "MI": "Mid-Level",
    "SE": "Senior-Level",
    "EX": "Executive-Level",
}

EMPLOYMENT_LABELS = {
    "FT": "Full-Time",
    "PT": "Part-Time",
    "CT": "Contract",
    "FL": "Freelance",
}

REMOTE_LABELS = {
    0: "On-Site (0%)",
    50: "Hybrid (50%)",
    100: "Fully Remote (100%)",
}

COUNTRY_LABELS = {
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "DE": "Germany",
    "IN": "India",
    "FR": "France",
    "ES": "Spain",
}

COMPANY_SIZE_LABELS = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}


# ---------------------------------------------------------
# Employee profile form
# ---------------------------------------------------------

st.subheader("Employee Profile")

st.write(
    "Select the employee and company characteristics that will be "
    "used to generate the salary estimate."
)

# Dividing the form into two columns reduces vertical scrolling and
# creates a cleaner interface.

left_column, right_column = st.columns(
    2,
    gap="large",
)


# ---------------------------------------------------------
# Left-side inputs
# ---------------------------------------------------------

with left_column:

    work_year = st.selectbox(
        "Work Year",
        options=[2020, 2021, 2022],
        index=2,
    )

    experience_level = st.selectbox(
        "Experience Level",
        options=list(EXPERIENCE_LABELS.keys()),
        index=2,
        format_func=lambda value: EXPERIENCE_LABELS[value],
    )

    employment_type = st.selectbox(
        "Employment Type",
        options=list(EMPLOYMENT_LABELS.keys()),
        index=0,
        format_func=lambda value: EMPLOYMENT_LABELS[value],
    )

    remote_ratio = st.selectbox(
        "Remote Work Ratio",
        options=list(REMOTE_LABELS.keys()),
        index=2,
        format_func=lambda value: REMOTE_LABELS[value],
    )


# ---------------------------------------------------------
# Right-side inputs
# ---------------------------------------------------------

with right_column:

    job_title = st.selectbox(
        "Job Title",
        options=[
            "Data Analyst",
            "Data Engineer",
            "Data Scientist",
            "Machine Learning Engineer",
            "Research Scientist",
            "Analytics Engineer",
            "Data Architect",
            "Principal Data Scientist",
            "Applied Machine Learning Scientist",
        ],
        index=2,
    )

    employee_residence = st.selectbox(
        "Employee Residence",
        options=list(COUNTRY_LABELS.keys()),
        index=0,
        format_func=lambda value: COUNTRY_LABELS[value],
    )

    company_location = st.selectbox(
        "Company Location",
        options=list(COUNTRY_LABELS.keys()),
        index=0,
        format_func=lambda value: COUNTRY_LABELS[value],
    )

    company_size = st.selectbox(
        "Company Size",
        options=list(COMPANY_SIZE_LABELS.keys()),
        index=1,
        format_func=lambda value: COMPANY_SIZE_LABELS[value],
    )


# ---------------------------------------------------------
# Generate the salary prediction
# ---------------------------------------------------------

st.markdown("---")

if st.button(
    "Predict Salary",
    type="primary",
):

    try:
        # Store the selected values in the same original-column format
        # used before feature encoding during model development.
        new_employee = pd.DataFrame(
            [
                {
                    "work_year": work_year,
                    "experience_level": experience_level,
                    "employment_type": employment_type,
                    "job_title": job_title,
                    "employee_residence": employee_residence,
                    "remote_ratio": remote_ratio,
                    "company_location": company_location,
                    "company_size": company_size,
                }
            ]
        )

        # Convert categorical values into numerical binary columns.
        new_employee_encoded = pd.get_dummies(
            new_employee,
            drop_first=False,
        )

        # Align the new record with the model's exact training features.
        # Features not represented by the selected employee receive zero.
        new_employee_encoded = new_employee_encoded.reindex(
            columns=feature_columns,
            fill_value=0,
        )

        # Generate the annual salary estimate.
        predicted_salary = salary_model.predict(
            new_employee_encoded
        )[0]

        # Prevent an unexpected negative number from being displayed.
        predicted_salary = max(
            float(predicted_salary),
            0.0,
        )

        # Display the prediction using a custom OU-themed result card.
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Estimated Annual Salary
                </div>
                <p class="prediction-value">
                    ${predicted_salary:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # Prediction summary
        # -------------------------------------------------
        # Native Streamlit containers are used instead of custom HTML.
        # This avoids the HTML-rendering problem that occurred earlier.

        st.subheader("Prediction Summary")

        summary_column_1, summary_column_2 = st.columns(
            2,
            gap="large",
        )

        with summary_column_1:

            with st.container(border=True):
                st.markdown("### Employee Information")

                st.write(f"**Work Year:** {work_year}")

                st.write(
                    "**Experience Level:** "
                    f"{EXPERIENCE_LABELS[experience_level]}"
                )

                st.write(
                    "**Employment Type:** "
                    f"{EMPLOYMENT_LABELS[employment_type]}"
                )

                st.write(
                    f"**Job Title:** {job_title}"
                )

        with summary_column_2:

            with st.container(border=True):
                st.markdown("### Location and Company Information")

                st.write(
                    "**Employee Residence:** "
                    f"{COUNTRY_LABELS[employee_residence]}"
                )

                st.write(
                    "**Remote Work:** "
                    f"{REMOTE_LABELS[remote_ratio]}"
                )

                st.write(
                    "**Company Location:** "
                    f"{COUNTRY_LABELS[company_location]}"
                )

                st.write(
                    "**Company Size:** "
                    f"{COMPANY_SIZE_LABELS[company_size]}"
                )

        st.caption(
            "Prediction generated from the trained Random Forest model. "
            "Results may vary because some compensation factors are not "
            "represented in the current dataset."
        )

        # Reinforce that the prediction is an estimate and not a guarantee.
        st.info(
            "This estimate is based on historical salary patterns and "
            "should be used as a decision-support tool rather than an "
            "exact salary recommendation."
        )

    except (ValueError, TypeError, KeyError) as error:
        st.error(
            "The salary prediction could not be generated. "
            "Please verify the selected employee information and try again."
        )
        st.exception(error)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Career Compass | Team DataForge | University of Oklahoma | "
    "Applied Machine Learning"
)
