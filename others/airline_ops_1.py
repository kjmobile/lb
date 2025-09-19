# %% [markdown]
# # Flight Delay Analysis (Intro Assignment)
#
# This notebook connects to the airline_db using pymysql, loads the flights_delay table,
# and performs two basic analyses:
# 1. Show the first few rows of the data
# 2. Calculate and print the average delay_minutes

# %%
import pymysql
import pandas as pd

# Connect to airline_db
connection = pymysql.connect(
    host='database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com',
    user='erau',
    password='1212',
    database='airline_db',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

print("Connected to database!")

# Load flights_delay table into pandas DataFrame
cursor = connection.cursor()
cursor.execute("SELECT * FROM flights_delay;")
rows = cursor.fetchall()
df_delay = pd.DataFrame(rows)

# Close connection
connection.close()

# %%
# Show the first few rows
print(df_delay.head())

# %%
# Calculate and print the average delay_minutes
average_delay = df_delay['delay_minutes'].mean()
print("Average delay (minutes):", average_delay)
