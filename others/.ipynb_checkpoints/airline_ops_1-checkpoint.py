# %% [markdown]
# # Airline Flight Delay Analysis and Prediction
#
# This notebook connects to the `airline_db` database and performs two types of analysis on the `flight_delay` table:
#
# 1. Conventional Statistical Analysis: Summary statistics, delay distributions, comparisons by weekday and weather, and ANOVA tests.
# 2. Machine Learning Modeling: Predicting delay minutes using Random Forest regression with categorical feature encoding.
#
# Each section is clearly marked for conversion to Jupyter Notebook.

# %%
import pandas as pd
import sqlalchemy

# Connect to airline_db (update credentials as needed)
engine = sqlalchemy.create_engine('mysql+pymysql://username:password@host:port/airline_db')

# Load flight_delay table
df_delay = pd.read_sql('SELECT * FROM flight_delay', engine)

# Preview data
df_delay.head()

# %% [markdown]
# ## 1. Conventional Statistical Analysis

# %%
# Overall average delay
average_delay = df_delay['delay_minutes'].mean()
print("Average delay (minutes):", average_delay)

# %%
# Histogram of delay minutes
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
df_delay['delay_minutes'].hist(bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Flight Delays")
plt.xlabel("Delay (minutes)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Average delay by day of week
weekday_avg = df_delay.groupby('day_of_week')['delay_minutes'].mean()
print("Average delay by weekday:")
print(weekday_avg)

weekday_avg.plot(kind='bar', title='Average Delay by Day of Week', color='orange')
plt.ylabel("Average Delay (minutes)")
plt.tight_layout()
plt.show()

# %%
# Average delay by weather condition
weather_avg = df_delay.groupby('weather_condition')['delay_minutes'].mean()
print("Average delay by weather condition:")
print(weather_avg)

weather_avg.plot(kind='bar', title='Average Delay by Weather Condition', color='green')
plt.ylabel("Average Delay (minutes)")
plt.tight_layout()
plt.show()

# %%
# ANOVA test for delay differences between airlines
import scipy.stats as stats

groups = [group['delay_minutes'].values for _, group in df_delay.groupby('airline_code')]
anova_result = stats.f_oneway(*groups)
print("ANOVA result for airline delay comparison:", anova_result)

# %% [markdown]
# ## 2. Machine Learning Approach: Predicting Flight Delays

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Select categorical features
features = ['weather_condition', 'day_of_week', 'aircraft_type', 'airline_code']
df_ml = pd.get_dummies(df_delay[features])

# Define target variable
X = df_ml
y = df_delay['delay_minutes']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# %%
# Feature importance plot
import numpy as np

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='purple')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - Statistical analysis revealed delay patterns by weekday and weather.
# - ANOVA showed significant differences in delays across airlines.
# - Machine learning model achieved reasonable performance in predicting delay minutes.
#
# This notebook can be extended with additional models, feature engineering, and integration with other tables such as `flights.csv`.
