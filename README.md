# Career Compass
### Employee Salary Prediction Using Machine Learning

Career Compass is a machine learning project developed as part of the University of Oklahoma Applied Computing graduate program. The objective of this project is to predict employee salaries within the data science industry using historical salary data and supervised machine learning techniques.

The project follows the Team Data Science Process (TDSP) by progressing through data acquisition, data preparation, exploratory data analysis, feature engineering, model development, evaluation, and deployment preparation.

---

## Project Objective

The goal of Career Compass is to develop a predictive model capable of estimating employee salaries based on employment characteristics such as:

- Experience Level
- Employment Type
- Job Title
- Employee Residence
- Remote Work Ratio
- Company Location
- Company Size
- Work Year

The final solution will be deployed through an interactive Streamlit web application that allows users to enter employee information and receive an estimated annual salary prediction.

---

## Dataset

**Dataset Name**

Global Data Science Salaries Dataset

**Source**

Kaggle Public Dataset

The dataset contains salary information for data science and machine learning professionals from multiple countries and organizations.

---

## Technologies Used

- Python 3.13
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Joblib
- Jupyter Notebooks
- Visual Studio Code
- Git & GitHub
- Streamlit (Future Deliverable)

---

## Project Structure

```text
CareerCompass/
├── app/
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

The project currently follows the workflow below:

1. Data Collection
2. Data Inspection
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Engineering
6. Model Training
7. Model Evaluation
8. Salary Prediction
9. Streamlit Deployment (Future Work)

---

## Current Model Performance

The current best-performing model is the **Random Forest Regressor**.

Performance on the testing dataset:

| Metric | Score |
|---------|---------|
| MAE | **$27,593.94** |
| RMSE | **$42,695.68** |
| R² | **0.5244** |

The Random Forest model outperformed both Linear Regression and Decision Tree Regression across all evaluation metrics.

---

## Current Repository Status

✔ Data Acquisition

✔ Data Cleaning

✔ Exploratory Data Analysis

✔ Feature Engineering

✔ Model Training

✔ Model Evaluation

✔ Prediction Demonstration

⬜ Streamlit Deployment

⬜ Model Optimization

⬜ Final Presentation

---

## Team

**Team DataForge**

- Lane Hodge
- Gary Malone
- Brooks Booker

University of Oklahoma

Master of Applied Computing Science

Machine Learning Course - Summer 2026

---

## Future Enhancements

Future project deliverables will include:

- Hyperparameter tuning
- Cross-validation improvements
- Streamlit web application
- Interactive salary prediction interface
- Additional feature engineering
- Model deployment
- Expanded documentation

---

## License

This repository was created for educational purposes as part of the University of Oklahoma Applied Computing graduate program.