#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Assignment: Multiple Regression for Flight Delay Prediction (Final Version)
# ## Objective
# The goal is to build a multiple regression model to predict flight arrival delays using two independent variables: departure delay and weather. We will learn how to use the formula feature in `statsmodels` to explicitly set a baseline for a categorical variable ('Weather').
# #
# ### Steps:
# 1.  **Connect to Database & Engineer Variables:** Write an SQL query to calculate `departure_delay` and retrieve `weather_condition` data.
# 2.  **Data Cleaning:** Remove null values and extreme outliers to improve model stability.
# 3.  **Build the Multiple Regression Model:** Create the model using `departure_delay` and `weather_condition` as independent variables, setting 'Clear' as the baseline for weather.
# 4.  **Analyze the Results:** Check the model's statistical significance and the impact of each variable on arrival delay.
# 5.  **Visualize the Results:** Use `seaborn` to visually analyze the model's performance and the effect of weather.
# 

# %%
# Step 0: Import necessary libraries
import pymysql
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Hide warning messages for cleaner output


# %% [markdown]
# ## Step 1: Connect to DB and Load Data
# We connect to the database and load data from the flights_delay table which contains all the engineered features we need.
# 

# %%
# Create a direct connection to the database
conn = pymysql.connect(
    host='database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com',
    user='erau',
    password='1212',
    database='airline_db',
    charset='utf8mb4'
)

# Query data from the flights_delay table
query = """
SELECT
    flight_id,
    departure_delay_mins,
    delay_minutes AS arrival_delay_mins,
    weather_condition,
    day_of_week,
    aircraft_type,
    airline_code
FROM
    flights_delay
WHERE
    departure_delay_mins IS NOT NULL
    AND delay_minutes IS NOT NULL
    AND weather_condition IS NOT NULL;
"""

# Execute the query and load data into a DataFrame
df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# %%
df.head()

# %% [markdown]
# ## Step 2: Data Cleaning and Preparation
# We clean the data by removing any missing values and filtering out extreme outliers, which could negatively impact the model's predictive power.
# 

# %%
# Check for missing values before cleaning
print(df.isnull().sum())

# %%
# Only drop rows with missing values in critical columns
df_cleaned = df.dropna(subset=['departure_delay_mins', 'arrival_delay_mins', 'weather_condition'])
# Use cleaned dataset for further analysis
df_cleaned

# %% [markdown]
# ## Step 3: Build the Multiple Regression Model
# Build the multiple regression model using `smf.ols`. 
# 
# dv: arrival_delay_mins ivs: departure_delay_mins,weather_condition
# 
# Inside the formula, Use the syntax `C(weather_condition, Treatment(reference='Clear'))` to explicitly set 'Clear' as the baseline category for weather.
# 

# %%
# Build the model using 'Clear' as the reference category (note the capital C)

model=smf.ols("arrival_delay_mins ~ departure_delay_mins + C(weather_condition, Treatment(reference='Clear'))", data=df_cleaned)
results= model.fit()

# %% [markdown]
# ## Step 4: Review and Interpret the Results
# The model summary will show the coefficient for `departure_delay_mins` as well as the additional impact of other weather conditions compared to 'Clear' weather.
# 

# %%
# Print the regression results summary

results.summary()


# %% [markdown]
# ## Step 5: Visualize the Results
# We analyze the model with two visualizations:
# 1.  A scatter plot comparing the actual and predicted values to check the overall predictive power of the model.
# 2.  A bar chart comparing the average arrival delay for each weather condition.
# 

# %%
# 1. Visualize the overall model fit (Actual vs. Predicted)
df_cleaned['predicted_delay'] = results.fittedvalues

plt.figure(figsize=(10, 6))
sns.scatterplot(x='predicted_delay', y='arrival_delay_mins', data=df_cleaned)
plt.plot([df_cleaned['arrival_delay_mins'].min(), df_cleaned['arrival_delay_mins'].max()],
         [df_cleaned['arrival_delay_mins'].min(), df_cleaned['arrival_delay_mins'].max()],'r--', lw=2)


# %%
# 2. Visualize the effect of weather condition on arrival delay
plt.figure(figsize=(10, 3))
weather_means = df_cleaned.groupby('weather_condition')['arrival_delay_mins'].mean().sort_values()
sns.barplot(x=weather_means.index, y=weather_means.values)
plt.title('Average Arrival Delay by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Average Arrival Delay (minutes)')


# %%
# Print summary statistics by weather condition
df_cleaned.groupby('weather_condition')['arrival_delay_mins'].describe()

# %%

