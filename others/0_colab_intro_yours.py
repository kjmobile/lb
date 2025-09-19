#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# <a href="https://colab.research.google.com/github/kjmobile/lb/blob/main/0_colab_intro_yours.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # I COLAB INTRO

# %% [markdown]
# # 0. Connect Database to get the dataset

# %%
! pip install pymysql

# %%
# import data from mySQL database  using the following info host: database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com, id erau, password 1212, db='data', port 3306
import pymysql
import pandas as pd

# %%
# 1.1 Practice DB connection
connection = pymysql.connect(
    host='database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com',
    user='erau',
    password='1212',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

cursor = connection.cursor()
print("Connected to database!")

# %%
#1.2 show available databases
cursor.execute("SHOW DATABASES")
print("Available databases:")
for db in cursor.fetchall():
    print(f"- {db['Database']}")

# %%
#1.3 airline_db select airline_db by USE commend
cursor.execute("USE airline_db")
print('connected to airline_db')

# %%
#1.3 airline_db 
cursor.execute("SHOW TABLES")
print("\nAvailable tables in airline_db:")
tables = cursor.fetchall()
for table in tables:
    table_name = list(table.values())[0]
    print(f"- {table_name}")

# %%
#1.4 select * from airports table.
cursor.execute("SELECT * FROM airports  WHERE city in ('Paris', 'Tokyo') ")
results =pd.DataFrame(cursor.fetchall())
results

# %%
#1.4 select * from airports table.
cursor.execute("""
SELECT f.flight_number,  a.airline_name
FROM flights f
JOIN airlines a ON f.airline_code = a.airline_code 
""")
results =pd.DataFrame(cursor.fetchall())
results

# %%
cursor.execute('''
SELECT * FROM bookings ORDER BY total_price LIMIT 1
''')
results=pd.DataFrame(cursor.fetchall())
results


# %%
#1.5 database connection close and cursor close
cursor.close()
connection.close()

# %% [markdown]
# ---

# %%
# 2 Perform compensation database with a single connection
with pymysql.connect(host='database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com',
                    user='erau',
                    password='1212',
                    db='hr',
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor) as connection:

    with connection.cursor() as cursor:
        # 1. Show available databases
        cursor.execute("SHOW DATABASES")
        print("Available databases:")
        for db in cursor.fetchall():
            print(db['Database'])

        print("\nAvailable tables in selected database:")
        
        # 2. Show tables in current database
        cursor.execute("SHOW TABLES")
        for table in cursor.fetchall():
            print(list(table.values())[0])  

        print("\nFetching compensation:")
        # 3. Fetch Salary_Data and create DataFrame
        cursor.execute("SELECT * FROM compensation")
        df = pd.DataFrame(cursor.fetchall())

# Display the DataFrame
df

# %% [markdown]
# # 1. Conventional Approach of Reression

# %% [markdown]
# ## 1.1 Check the Dataset

# %%
df.head()

# %%
import seaborn as sns
sns.regplot(data=df, x='YearsExperience', y='Salary', line_kws={'color': 'red', 'lw':.5})

# %% [markdown]
# ## 1.2 Fitting the Model

# %%
# dataframe df: run linear regression using statmodels

import statsmodels.formula.api as smf
# Fit the linear regression model
model = smf.ols(formula="Salary ~ YearsExperience", data=df).fit()

# Print the model summary
print(model.summary())


# %% [markdown]
# '''
# APA report example:
# '''
#     A linear regression analysis was conducted to examine the relationship between years of experience and salary.
#     The model explained a significant proportion of variance in salary, R² = .957, F(1, 28) = 622.5, p < .001.
# 
#     Years of experience significantly predicted salary (β = 9449.96, SE = 378.76, p < .001), with each additional
#     year of experience associated with an increase of $9,449.96 in salary. The model's intercept was $25,790
#     (SE = 2273.05, p < .001), representing the predicted salary for someone with no experience.
# 
#     
# 
#     The regression equation can be expressed as:
#     Salary = 25,790 + 9449.96 × YearsExperience
# '''

# %% [markdown]
# # 2. Machine Learning Approach

# %% [markdown]
# ## 2.1 Check Dataset

# %%
df.head()

# %% [markdown]
# ## 2.2 Preprocessing : Train-Test Split

# %%

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['YearsExperience']], df['Salary'], test_size=0.25, random_state=0)


# %% [markdown]
# ## 2.3 Model Fitting: Using Train Set

# %%

from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# %%
# check model coeffs

print(model.coef_.round(2))
print(model.intercept_.round(2))

# %%
model.coef_

# %%
model.intercept_

# %%
pd.Series(model.coef_, index = X_train.columns)

# %%
# specify model equation

print(f"Salary = 9869.5 * YearsExperience + 25792.2")

# %% [markdown]
# ## 2.4 Make prediction : using Test set

# %%
# Make predictions using the model
y_pred = model.predict(X_test)


# %%
y_pred

# %%
# Plot the actual and predicted values
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='red', label='Actual')
plt.plot(X_test, y_pred, color='blue', label='Predicted')
plt.title('Actual vs. Predicted Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.legend()



# %% [markdown]
# ## 2.5 Evaluate the Model : $R^2$ and $MSE$ mean squared error
# - MSE(or RMSE) is a 'relative metric' : used to compare with other models

# %%
model.score(X_test, y_test)

# %%
# evaluate the model

from sklearn.metrics import mean_squared_error

# Calculate the mean squared error
mse = round(mean_squared_error(y_test, y_pred), 2)

# Print the mean squared error
print("Mean squared error:", mse)


# %% [markdown]
# ---
# ---

# %% [markdown]
# ## 2.5 Some Equations

# %% [markdown]
# ### 2.5.1 **Coe-efficient of Determination**

# %% [markdown]
# $$ R^2 = 1−\frac{\text{Residual Sum of Squares (RSS)}}{\text{Total Sum of Square (TSS)}​} $$  

# %%


# %% [markdown]
# 
# $$ \text{R}^2 =1-\frac{\sum\limits_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum\limits_{i=1}^{N} (y_i - \bar{y})^2}\ $$  
# 

# %% [markdown]
# <img src="https://github.com/kjmobile/data/blob/main/img/R_sq.png?raw=true" width=700 />

# %% [markdown]
# ### 2.5.2 **Mean Squared Error (MSE)** :
# 

# %% [markdown]
# 
# $$
# \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
# $$
# 

# %% [markdown]
# 
# **note:**
# 
# * $\frac{1}{n}$: This is the normalization factor dividing by the number of observations.
# * $\sum_{i=1}^{n}$: This is the summation symbol, indicating we sum over all observations from 1 to n.
# * $y_i$: This represents the actual value of the i-th observation.
# * $\hat{y}_i$: This represents the predicted value of the i-th observation.
# * $^2$: This squares the difference between the actual and predicted values.
# 
# 
