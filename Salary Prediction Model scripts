
#SALARY PREDICTION

#Building a salary prediction model that predicts the salary according to features 
# Data source is : "https://www.kaggle.com/datasets/mrsimple07/salary-prediction-data"

#Education: Categorical variable indicating the highest level of education attained by an individual (categorical).
#Experience: The integer variable representing the number of years of professional experience (integer).
#Location: The geographical location of employment, categorized as 'Urban', 'Suburban', or 'Rural' (categorical).
#Job_Title: The job title or position held by an individual (categorical).
#Age: The age of the individual (integer).
#Gender: The gender of the individual (categorical).
#Salary: The salary of the individual (float).

#Project Structure:
# 1. Importing the libraries
# 2. Data Loading and Cleaning
# 3. Enhanced Exploratory Data Analysis (EDA) and Visualization
# 4. Future engineering and Preprocessing for Modeling
# 5. Model Building and Evaluation
# 6. Model Saving

#Importing the libraries: 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from joblib import dump
from xgboost import XGBRegressor

# Data loading and cleaning
salary_df = pd.read_csv('salary_prediction_data.csv')
salary_df.dropna(inplace=True)
salary_df

# Enhanced Exploratory Data Analysis (EDA)
print(salary_df.info())  # Data overview
print(salary_df.describe())

# Heatmap for correlation
numeric_df = salary_df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Salary Distribution
sns.histplot(salary_df['Salary'], kde=True)
plt.title('Salary Distribution')
plt.show()

# Salary Distribution by each feature
for column in salary_df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=column, y='Salary', data=salary_df)
    plt.xticks(rotation=45)
    plt.title(f'Salary Distribution by {column}')
    plt.show()

## Feature Engineering
categorical_cols = [cname for cname in salary_df.columns if salary_df[cname].dtype == "object"]
numerical_cols = [cname for cname in salary_df.columns if salary_df[cname].dtype in ['int64', 'float64'] and cname != 'Salary']

poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly_transformer.fit_transform(salary_df[numerical_cols])
poly_feature_names = poly_transformer.get_feature_names_out(input_features=numerical_cols)
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

# Concatenating the original DataFrame with the new polynomial features DataFrame
salary_df_extended = pd.concat([salary_df.drop(columns=numerical_cols).reset_index(drop=True), poly_df], axis=1)
salary_df_extended
# Preprocessing
numerical_cols_extended = list(poly_feature_names)  # Updated to prevent duplication
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())]), numerical_cols_extended),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

# Model Experimentation and Hyperparameter Tuning
X = salary_df_extended.drop('Salary', axis=1)
y = salary_df_extended['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# RandomForestRegressor Model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=0))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # Ensuring that predictions are generated from the model
print(f'Random Forest MAE: {mean_absolute_error(y_test, predictions)}')

# Grid Search for Gradient Boosting
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=0))
])
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.01, 0.1]
}
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
print(f'Best parameters for Gradient Boosting: {grid_search.best_params_}')
predictions_best = grid_search.best_estimator_.predict(X_test)
print(f'Gradient Boosting MAE: {mean_absolute_error(y_test, predictions_best)}')

# Additional Evaluation Metrics
mse = mean_squared_error(y_test, predictions_best)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions_best)
print(f'MSE: {mse}, RMSE: {rmse}, R^2: {r2}')

# Saving the best model
dump(grid_search.best_estimator_, 'salary_prediction_best_model.joblib')