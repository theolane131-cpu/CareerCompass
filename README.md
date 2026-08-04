# Career Compass

### Employee Salary Prediction Using Machine Learning

Career Compass is a machine learning project developed as part of the University of Oklahoma Applied Computing graduate program. The project predicts employee salaries within the data science industry using historical salary data and supervised machine learning techniques.

The project follows the Team Data Science Process (TDSP) by progressing through data acquisition, data preparation, exploratory data analysis, feature engineering, model development, evaluation, model persistence, and interactive application development.

---

## Project Objective

The goal of Career Compass is to estimate an employee's annual salary based on employment and organizational characteristics such as:

- Experience Level
- Employment Type
- Job Title
- Employee Residence
- Remote Work Ratio
- Company Location
- Company Size
- Work Year

The completed minimum viable product uses an interactive Streamlit application that allows users to select employee characteristics and receive an estimated annual salary from the trained Random Forest model.

The prediction is intended to support salary benchmarking and compensation analysis. It should be interpreted as an informed estimate rather than an exact compensation recommendation.

---

## Dataset

**Dataset Name**

Global Data Science Salaries Dataset

**Source**

Kaggle Public Dataset

The dataset contains salary information for data science and machine learning professionals from multiple countries and organizations. It includes employee, job, location, company, remote-work, and compensation attributes.

The target variable used for machine learning is:

```text
salary_in_usd
```

The original `salary` and `salary_currency` fields were excluded from the model inputs to prevent data leakage.

---

## Technologies Used

- Python 3.13
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebooks
- Visual Studio Code
- Git and GitHub

---

## Project Structure

```text
CareerCompass/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── ds_salaries.csv
│   └── processed/
│       ├── processed_salary_data.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   └── random_forest_salary_model.joblib
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training_and_evaluation.ipynb
│   └── 06_salary_prediction_demo.ipynb
├── reports/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Machine Learning Workflow

The Career Compass workflow includes:

1. Data Collection
2. Data Inspection
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. One-Hot Encoding
7. Train/Test Split
8. Model Training
9. Model Evaluation
10. Model Selection
11. Model Persistence with Joblib
12. Salary Prediction Demonstration
13. Streamlit MVP Development

Eight original predictor variables were expanded into 164 encoded features through one-hot encoding. This allowed categorical variables to be used by the regression models without assigning an artificial numerical order to their values.

---

## Models Evaluated

Three supervised regression models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

The models were evaluated using:

- Mean Absolute Error
- Root Mean Squared Error
- Coefficient of Determination

The Random Forest Regressor produced the strongest overall results and was selected as the final model.

---

## Current Model Performance

Performance of the selected Random Forest model on the testing dataset:

| Metric | Score |
|---|---:|
| MAE | **$27,593.94** |
| RMSE | **$42,695.68** |
| R² | **0.5244** |

The Mean Absolute Error indicates that predictions differ from actual salaries by approximately $27,600 on average. The higher RMSE indicates that some larger prediction errors remain, particularly for uncommon high-salary observations. The R² score shows that the model explains approximately 52.4% of the variation in employee salaries.

---

## Streamlit MVP

Career Compass includes a working Streamlit minimum viable product that loads the saved Random Forest model and generates salary predictions from user-selected employee and company characteristics.

### Application Inputs

The application accepts:

- Work Year
- Experience Level
- Employment Type
- Job Title
- Employee Residence
- Remote Work Ratio
- Company Location
- Company Size

### Application Output

The application returns:

- Estimated Annual Salary
- Employee Profile Summary
- Location and Company Summary
- Prediction Disclaimer

The interface uses an OU-inspired crimson theme and a soft ivory background to provide a clean and consistent user experience.

### Run the Application Locally

From the project root, activate the project's virtual environment and install the required packages:

```bash
python -m pip install -r requirements.txt
```

Then start the Streamlit application:

```bash
python -m streamlit run app/app.py
```

Streamlit will display a local address, typically:

```text
http://localhost:8501
```

Open that address in a web browser to use the application.

### Example Prediction

A senior-level, full-time Data Scientist residing in the United States, working fully remotely for a medium-sized U.S. company in 2022, received an estimated annual salary of approximately:

```text
$168,265.00
```

The prediction is based on historical patterns in the training dataset and may not reflect every factor that influences real-world compensation.

---

## Current Repository Status

- [x] Data Acquisition
- [x] Data Inspection
- [x] Data Cleaning
- [x] Exploratory Data Analysis
- [x] Feature Engineering
- [x] One-Hot Encoding
- [x] Model Training
- [x] Model Evaluation
- [x] Model Selection
- [x] Prediction Demonstration
- [x] Model Persistence
- [x] GitHub Version Control
- [x] Streamlit MVP
- [x] Final Project Presentation

---

## Model and Application Limitations

The current project has several limitations:

- The dataset contains only 607 records.
- Salary records cover 2020 through 2022.
- Some job titles, countries, and employment categories have small sample sizes.
- The model focuses on data science and related technology roles.
- Important factors such as education, certifications, exact years of experience, benefits, negotiation, industry, and local cost of living are not included.
- Prediction accuracy may be weaker for categories with limited representation in the training dataset.

Career Compass should therefore be used as a decision-support tool rather than as a source of guaranteed compensation values.

---

## Team

**Team DataForge**

- Lane Hodge
- Gary Malone
- Brooks Booker

University of Oklahoma  
Master of Applied Computing Science  
Machine Learning Course — Summer 2026

---

## Future Enhancements

Potential future improvements include:

- Hyperparameter tuning for the Random Forest model
- Comparison with additional ensemble and boosting models
- Expanded and more recent salary datasets
- Additional employee and company features
- Model performance monitoring
- Improved support for underrepresented countries and job titles
- Additional application testing and accessibility refinement

---

## License

This repository was created for educational purposes as part of the University of Oklahoma Applied Computing graduate program.