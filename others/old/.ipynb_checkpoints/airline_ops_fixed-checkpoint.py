# %% [markdown]
# # Assignment: Multiple Regression for Flight Delay Prediction (Final Version)
# ## Objective
# The goal is to build a multiple regression model to predict flight arrival delays using two independent variables: departure delay and weather. We will learn how to use the formula feature in `statsmodels` to explicitly set a baseline for a categorical variable ('Weather').
#
# ### Steps:
# 1.  **Connect to Database & Engineer Variables:** Write an SQL query to calculate `departure_delay` and retrieve `weather_condition` data.
# 2.  **Data Cleaning:** Remove null values and extreme outliers to improve model stability.
# 3.  **Build the Multiple Regression Model:** Create the model using `departure_delay` and `weather_condition` as independent variables, setting 'Clear' as the baseline for weather.
# 4.  **Analyze the Results:** Check the model's statistical significance and the impact of each variable on arrival delay.
# 5.  **Visualize the Results:** Use `seaborn` to visually analyze the model's performance and the effect of weather.

# %%
# Step 0: Import necessary libraries
import pymysql
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

print("Libraries imported successfully.")

# %% [markdown]
# ## Step 1: Connect to DB and Engineer Features
# We will execute an SQL query to load and create the variables needed for the model. We use the `TIMESTAMPDIFF()` function to calculate departure delay and select the `weather_condition` column.

# %%
# Create a direct connection to the database
conn = pymysql.connect(
    host='database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com',
    user='erau',
    password='1212',
    database='airline_db',
    charset='utf8mb4'
)

# First, check what data we have in the tables
check_query = """
SELECT COUNT(*) as flight_count FROM flights;
"""
count_df = pd.read_sql_query(check_query, conn)
print("Total flights in database:", count_df['flight_count'].values[0])

check_query2 = """
SELECT COUNT(*) as delay_count FROM flights_delay;
"""
count_df2 = pd.read_sql_query(check_query2, conn)
print("Total records in flights_delay:", count_df2['delay_count'].values[0])

# Check status values
status_query = """
SELECT status, COUNT(*) as count 
FROM flights 
GROUP BY status;
"""
status_df = pd.read_sql_query(status_query, conn)
print("\nFlight status distribution:")
print(status_df)

# Modified query - remove WHERE clause to see if we get any data
feature_engineering_query = """
SELECT
    TIMESTAMPDIFF(MINUTE, f.scheduled_departure, f.actual_departure) AS departure_delay_mins,
    fd.weather_condition,
    fd.delay_minutes AS arrival_delay_mins
FROM
    flights AS f
JOIN
    flights_delay AS fd ON f.flight_id = fd.flight_id
LIMIT 1000;
"""

# Execute the query and load data into a DataFrame
df = pd.read_sql_query(feature_engineering_query, conn)

# If still no data, try a simpler query
if len(df) == 0:
    print("\nNo data from JOIN. Trying simpler query...")
    simple_query = """
    SELECT 
        flight_id,
        delay_minutes,
        weather_condition
    FROM flights_delay
    LIMIT 100;
    """
    df_simple = pd.read_sql_query(simple_query, conn)
    print("Sample from flights_delay table:")
    print(df_simple.head())

# Close the database connection
conn.close()
print("\nDatabase connection closed.")
print("Data shape:", df.shape)
print(df.head())


# %% [markdown]
# ## Step 2: Data Cleaning and Preparation
# We clean the data by removing any missing values and filtering out extreme outliers, which could negatively impact the model's predictive power.

# %%
# Drop any rows with missing values
df.dropna(inplace=True)

print("\nData cleaned. Shape of the final dataset:", df.shape)


# %% [markdown]
# ## Step 3: Build the Multiple Regression Model
# We build the multiple regression model using `smf.ols`. Inside the formula, we use the syntax `C(weather_condition, Treatment(reference='clear'))` to explicitly set 'clear' as the baseline category for weather.

# %%
# Check weather_condition data first
print("Unique weather conditions:", df['weather_condition'].unique())
print("Weather condition counts:")
print(df['weather_condition'].value_counts())

# Build the model - if weather data has issues, use simple regression
if df['weather_condition'].nunique() > 1:
    # Use lowercase 'clear' as the reference category
    formula = "arrival_delay_mins ~ departure_delay_mins + C(weather_condition, Treatment(reference='clear'))"
else:
    print("Warning: Only one weather condition found. Using simple linear regression.")
    formula = "arrival_delay_mins ~ departure_delay_mins"

# Create and fit the Ordinary Least Squares (OLS) model using the formula
model = smf.ols(formula=formula, data=df)
results = model.fit()

# %% [markdown]
# ## Step 4: Review and Interpret the Results
# The model summary will show the coefficient for `departure_delay_mins` as well as the additional impact of other weather conditions compared to 'clear' weather.

# %%
# Print the regression results summary
print(results.summary())


# %% [markdown]
# ## Step 5: Visualize the Results
# We analyze the model with two visualizations:
# 1.  A scatter plot comparing the actual and predicted values to check the overall predictive power of the model.
# 2.  A bar chart comparing the average arrival delay for each weather condition.

# %%
# 1. Visualize the overall model fit (Actual vs. Predicted)
df['predicted_delay'] = results.fittedvalues
sns.lmplot(x='predicted_delay', y='arrival_delay_mins', data=df, height=6, aspect=1.5,
           line_kws={'color': 'red'}, scatter_kws={'alpha': 0.3})
plt.title('Actual vs. Predicted Flight Delays (Multiple Regression)')
plt.xlabel('Predicted Delay (minutes)')
plt.ylabel('Actual Delay (minutes)')
plt.grid(True)
plt.show()

# 2. Visualize the effect of weather condition on arrival delay
print("\nGenerating plot for weather conditions...")
sns.catplot(x='weather_condition', y='arrival_delay_mins', data=df, kind='bar', 
            height=6, aspect=1.5, order=['clear', 'cloudy', 'rain', 'fog', 'snow', 'thunderstorm'])
plt.title('Average Arrival Delay by Weather Condition')
plt.xlabel('Weather Condition')
plt.ylabel('Average Arrival Delay (minutes)')
plt.grid(axis='y')
plt.show()