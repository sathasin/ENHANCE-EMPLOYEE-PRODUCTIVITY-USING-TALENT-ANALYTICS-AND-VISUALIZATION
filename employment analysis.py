# Importing necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
import xgboost as xgb

# Reading the dataset from a CSV file
df = pd.read_csv("/content/train_dataset.csv")

# Displaying the first few rows of the dataset
print(df.head())

# Displaying information about the dataset
print(df.info())

# Displaying descriptive statistics of the dataset
print(df.describe())

# Displaying the column names of the dataset
print(df.columns)

# Checking for missing values in the dataset
print(df.isna().sum())

# Handling missing values by filling NaN values in the 'wip' column with the mean
df['wip'] = df['wip'].fillna(df['wip'].mean())

# Plotting a histogram for the 'actual_productivity' column
sns.histplot(data=df['actual_productivity'], kde=False)
plt.show()

# Calculating the correlation matrix for the dataset
corr_matrix = df.corr()

# Creating a heatmap to visualize the correlation matrix
plt.figure(figsize=(30, 30))
sns.heatmap(corr_matrix, annot=True)
plt.show()

# Extracting the correlation of each feature with the target variable
corr_matrix_with_target = df.corr()['targeted_productivity'].sort_values(ascending=False)
print(corr_matrix_with_target)

# Separating the features (x) and the target variable (y)
x = df.drop(columns='targeted_productivity')
y = df['targeted_productivity']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# AdaBoost Regressor - Hyperparameter Tuning
param_grid_adaboost = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.2, 0.3]
}
adaboost_regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), random_state=42)
grid_search_adaboost = GridSearchCV(adaboost_regressor, param_grid=param_grid_adaboost, cv=5, scoring='neg_mean_squared_error')
grid_search_adaboost.fit(x_train, y_train)
best_adaboost = grid_search_adaboost.best_estimator_
ada_pred_tuned = best_adaboost.predict(x_test)
score_adaboost_tuned = r2_score(y_test, ada_pred_tuned)
print("Tuned AdaBoost Regressor R2 Score:", score_adaboost_tuned)


# Original XGBoost Regressor
param_dist_xgb = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
}
xgb_regressor = xgb.XGBRegressor()
random_search_xgb = RandomizedSearchCV(xgb_regressor, param_distributions=param_dist_xgb, n_iter=10, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1, random_state=42)
random_search_xgb.fit(x_train, y_train)
best_xgb = random_search_xgb.best_estimator_
xgb_pred = best_xgb.predict(x_test)
score_xgb = r2_score(y_test, xgb_pred)
print("Original XGBoost Regressor R2 Score:", score_xgb)
