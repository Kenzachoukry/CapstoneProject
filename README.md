# Salary Prediction Model

## Overview
This project aims to predict individual salaries based on various features using machine learning techniques. The model is built using a dataset available on Kaggle at [this link](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data). The dataset includes features such as education level, years of experience, job location, job title, age, gender, and the salary itself.

## Features
- **Education**: Categorical variable indicating the highest level of education attained by an individual.
- **Experience**: Integer variable representing the number of years of professional experience.
- **Location**: Categorical variable for the geographical location of employment, categorized as 'Urban', 'Suburban', or 'Rural'.
- **Job_Title**: Categorical variable for the job title or position held by an individual.
- **Age**: Integer variable for the age of the individual.
- **Gender**: Categorical variable for the gender of the individual.
- **Salary**: Float variable for the individual's salary.

## Project Structure
1. **Importing Libraries**: Essential Python libraries for data manipulation, visualization, and machine learning.
2. **Data Loading and Cleaning**: Loading the dataset and performing initial data cleaning operations.
3. **Exploratory Data Analysis (EDA) and Visualization**: Detailed exploration of the data to uncover relationships and trends.
4. **Feature Engineering and Preprocessing for Modeling**: Preparing the data for modeling by encoding categorical variables, normalizing numerical variables, and creating polynomial features.
5. **Model Building and Evaluation**: Building and evaluating machine learning models using techniques such as Random Forest and Gradient Boosting.
6. **Model Saving**: Saving the best-performing model for future predictions.

## Dependencies
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib

## Getting Started
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies: `pip install pandas seaborn matplotlib scikit-learn xgboost joblib`
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data) and place it in the project directory.
4. Run the notebook or script to train the model and make predictions.

## Model Performance
The performance of the model is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the coefficient of determination (R^2).

