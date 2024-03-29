#Education: The educational level of the person (categorical).
#Experience: The number of years of work experience (integer).
#Location: The work location (categorical).
#Job_Title: The title of the job (categorical).
#Age: The age of the person (integer).
#Gender: The gender of the person (categorical).
#Salary: The salary of the person (float).

#Project Structure
#Data Loading and Cleaning
#Exploratory Data Analysis (EDA) and Visualization
#Preprocessing for Modeling
#Model Building and Evaluation
#Model Saving

#Importing the libraries: 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Data loading and cleaning 
salary_df = pd.read_csv('salary_prediction_data.csv')
salary_df.dropna(inplace=True)
salary_df



# 2. Exploratory Data Analysis (EDA) and Visualization
# Getting a concise summary of the DataFrame
print(salary_df.info())

# Generating descriptive statistics
print(salary_df.describe())

# Listing all the columns
print(salary_df.columns)

# Checking for missing values
print(salary_df.isnull().sum())

#Correlation matrix 
# Correlation matrix heatmap
sns.heatmap(salary_df.corr(), annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#Salary distribution 
sns.histplot(salary_df['Salary'], kde=True)
plt.title('Salary Distribution')
plt.show()

# Categorical data distribution
for column in salary_df.select_dtypes(include=['object']).columns:
    sns.countplot(y=column, data=salary_df)
    plt.title(f'Distribution of {column}')
    plt.show()





#Checking the correlation between the variables: 
salary_df.corr()

#The columns of interest for this analysis include:
#Education   
#Experience  
#Location    
#Job_Title   
#Age          
#Gender       
#Salary 

import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()
Bar Charts for Categorical Data: To visualize the distribution of categories.

python
Copy code
data['Education'].value_counts().plot(kind='bar')
plt.show()
Scatter Plots: To visualize relationships between variables.

python
Copy code
data.plot(kind="scatter", x="Experience", y="Salary")
plt.show()


# Preprocessing
categorical_cols = [cname for cname in salary_df.columns if salary_df[cname].dtype == "object"]
numerical_cols = [cname for cname in salary_df.columns if salary_df[cname].dtype in ['int64', 'float64'] and cname != 'Salary']

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Separate target from predictors
y = data.Salary
X = data.drop(['Salary'], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = clf.predict(X_test)

print('MAE:', mean_absolute_error(y_test, preds))

# Save the model
dump(clf, 'salary_prediction_model.joblib')