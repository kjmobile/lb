# %% [markdown]
# # Flight Delay Data Generation and Database Update
# This notebook generates realistic flight delay data and updates the database

# %% [markdown]
# ## Step 1: Import Libraries and Define Functions

# %%
import pandas as pd
import numpy as np
import pymysql
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

print("Libraries imported successfully!")

# %% [markdown]
# ## Step 2: Define Database Functions

# %%
def alter_table_structure(conn):
    """Add departure_delay_mins column to flights_delay table"""
    try:
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'airline_db' 
            AND TABLE_NAME = 'flights_delay' 
            AND COLUMN_NAME = 'departure_delay_mins'
        """)
        
        if cursor.fetchone()[0] == 0:
            # Add the column if it doesn't exist
            cursor.execute("""
                ALTER TABLE flights_delay 
                ADD COLUMN departure_delay_mins INT AFTER delay_minutes
            """)
            conn.commit()
            print("✓ Added departure_delay_mins column to flights_delay table")
        else:
            print("✓ Column departure_delay_mins already exists")
        
        cursor.close()
        return True
        
    except Exception as e:
        print(f"Error altering table: {e}")
        return False

# %% [markdown]
# ## Step 3: Define Data Generation Function

# %%
def generate_realistic_flight_delay_data(n_samples=500):
    """
    Generate realistic flight delay data for multiple regression analysis
    """
    
    # Generate base features
    data = {
        'flight_id': range(1, n_samples + 1),
        'delay_id': range(1, n_samples + 1)
    }
    
    # Weather conditions with realistic distribution
    weather_conditions = ['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow', 'Thunderstorm']
    weather_probs = [0.45, 0.25, 0.15, 0.08, 0.04, 0.03]
    data['weather_condition'] = np.random.choice(weather_conditions, n_samples, p=weather_probs)
    
    # Weather impact factors
    weather_impact = {
        'Clear': 0,
        'Cloudy': 5,
        'Rain': 15,
        'Fog': 30,
        'Snow': 40,
        'Thunderstorm': 50
    }
    
    # Day of week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_traffic_factor = [1.2, 1.1, 1.0, 1.1, 1.3, 0.9, 1.0]
    data['day_of_week'] = np.random.choice(days, n_samples)
    
    # Aircraft types
    aircraft_types = ['A320-200', 'A321-200', 'A330-300', 'A350-900', 'A380-800', 
                     'B737-900', 'B737-MAX', 'B747-8', 'B777-300', 'B787-9']
    aircraft_turnaround = {
        'A320-200': 0, 'A321-200': 0, 'B737-900': 0, 'B737-MAX': 0,
        'A330-300': 5, 'A350-900': 5, 'B777-300': 5, 'B787-9': 5,
        'A380-800': 10, 'B747-8': 10
    }
    data['aircraft_type'] = np.random.choice(aircraft_types, n_samples)
    
    # Airlines
    airlines = ['KE', 'OZ', 'AA', 'DL', 'UA', 'JL', 'NH', 'SQ', 'BA', 'LH', 'AF', 'EK']
    airline_efficiency = {
        'SQ': -5, 'NH': -3, 'JL': -3,
        'KE': 0, 'OZ': 0, 'LH': 0, 'BA': 0,
        'AA': 3, 'DL': 3, 'UA': 3,
        'AF': 5, 'EK': 5
    }
    data['airline_code'] = np.random.choice(airlines, n_samples)
    
    # Generate departure delays
    base_departure_delays = np.random.exponential(scale=15, size=n_samples)
    base_departure_delays = np.where(np.random.random(n_samples) < 0.3, 0, base_departure_delays)
    base_departure_delays = np.clip(base_departure_delays, 0, 180)
    
    departure_delays = []
    for i in range(n_samples):
        weather_delay = weather_impact[data['weather_condition'][i]]
        airline_delay = airline_efficiency[data['airline_code'][i]]
        day_factor = day_traffic_factor[days.index(data['day_of_week'][i])]
        
        departure_delay = base_departure_delays[i] * day_factor + weather_delay * 0.5 + airline_delay
        departure_delay = max(0, departure_delay + np.random.normal(0, 5))
        departure_delays.append(round(departure_delay))
    
    data['departure_delay_mins'] = departure_delays
    
    # Generate arrival delays (correlated with departure delay)
    arrival_delays = []
    for i in range(n_samples):
        # Strong correlation with departure delay (coefficient ≈ 1.1)
        base_arrival = data['departure_delay_mins'][i] * 1.1
        
        # Add weather impact
        weather_delay = weather_impact[data['weather_condition'][i]]
        
        # Add aircraft turnaround impact
        aircraft_delay = aircraft_turnaround[data['aircraft_type'][i]]
        
        # Add airline efficiency
        airline_delay = airline_efficiency[data['airline_code'][i]]
        
        # Calculate total with some randomness
        total_arrival_delay = (
            base_arrival + 
            weather_delay * 0.7 +
            aircraft_delay +
            airline_delay +
            np.random.normal(0, 8)  # Random variation
        )
        
        # Ensure non-negative and round
        total_arrival_delay = max(0, round(total_arrival_delay))
        arrival_delays.append(total_arrival_delay)
    
    data['arrival_delay_mins'] = arrival_delays
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

print("Functions defined successfully!")

# %% [markdown]
# ## Step 4: Generate the Data

# %%
# Generate the data
print("Generating realistic flight delay data...")
df_delays = generate_realistic_flight_delay_data(500)

print(f"✓ Generated {len(df_delays)} records")

# %% [markdown]
# ## Step 5: Display Data Summary

# %%
# Display summary statistics
print("=" * 60)
print("DATA SUMMARY")
print("=" * 60)

print(f"\nDeparture delay statistics:")
print(f"  Mean: {df_delays['departure_delay_mins'].mean():.1f} minutes")
print(f"  Std:  {df_delays['departure_delay_mins'].std():.1f} minutes")
print(f"  Max:  {df_delays['departure_delay_mins'].max()} minutes")

print(f"\nArrival delay statistics:")
print(f"  Mean: {df_delays['arrival_delay_mins'].mean():.1f} minutes")
print(f"  Std:  {df_delays['arrival_delay_mins'].std():.1f} minutes")
print(f"  Max:  {df_delays['arrival_delay_mins'].max()} minutes")

# Calculate correlation
correlation = df_delays[['departure_delay_mins', 'arrival_delay_mins']].corr().iloc[0,1]
print(f"\nCorrelation between delays: {correlation:.3f}")

# %% [markdown]
# ## Step 6: Show Weather Distribution

# %%
print("Weather distribution:")
weather_counts = df_delays['weather_condition'].value_counts()
for weather, count in weather_counts.items():
    print(f"  {weather:12}: {count:3} ({count/len(df_delays)*100:.1f}%)")

# %% [markdown]
# ## Step 7: Display Sample Data

# %%
print("Sample Generated Data (first 10 rows):")
print(df_delays[['delay_id', 'departure_delay_mins', 'arrival_delay_mins', 
                 'weather_condition', 'airline_code']].head(10))

# %% [markdown]
# ## Step 8: Save to CSV

# %%
# Save to CSV for backup
csv_filename = 'flight_delays_enhanced.csv'
df_delays.to_csv(csv_filename, index=False)
print(f"✓ Data saved to '{csv_filename}'")

# %% [markdown]
# ## Step 9: Update Database (Optional)
# **WARNING**: This will DELETE all existing data in the flights_delay table!

# %%
# Ask for confirmation before updating database
update_db = input("Do you want to update the database? This will DELETE existing data! (yes/no): ")

if update_db.lower() == 'yes':
    try:
        # Create connection
        conn = pymysql.connect(
            host='database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com',
            user='erau',
            password='1212',
            database='airline_db',
            charset='utf8mb4'
        )
        
        print("\n✓ Connected to database")
        
        # Alter table structure
        if alter_table_structure(conn):
            
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM flights_delay")
            print(f"✓ Cleared existing data")
            
            # Insert new data
            insert_query = """
            INSERT INTO flights_delay 
            (delay_id, flight_id, delay_minutes, departure_delay_mins,
             weather_condition, day_of_week, aircraft_type, airline_code) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            inserted = 0
            for index, row in df_delays.iterrows():
                values = (
                    row['delay_id'],
                    row['flight_id'],
                    row['arrival_delay_mins'],
                    row['departure_delay_mins'],
                    row['weather_condition'],
                    row['day_of_week'],
                    row['aircraft_type'],
                    row['airline_code']
                )
                cursor.execute(insert_query, values)
                inserted += 1
                
                if inserted % 100 == 0:
                    print(f"  Inserted {inserted} records...")
            
            conn.commit()
            print(f"✓ Successfully inserted {inserted} records")
            
            # Verify
            cursor.execute("SELECT COUNT(*) FROM flights_delay")
            count = cursor.fetchone()[0]
            print(f"✓ Verified: {count} records in database")
            
            cursor.close()
            conn.close()
            print("✓ Database updated successfully!")
            
    except Exception as e:
        print(f"Error: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
else:
    print("Database update cancelled.")

# %% [markdown]
# ## Step 10: Quick Data Verification

# %%
# Quick check if data was inserted correctly
if update_db.lower() == 'yes':
    try:
        conn = pymysql.connect(
            host='database-klee.cbgcswckszgl.us-east-1.rds.amazonaws.com',
            user='erau',
            password='1212',
            database='airline_db',
            charset='utf8mb4'
        )
        
        query = """
        SELECT 
            delay_id,
            departure_delay_mins,
            delay_minutes as arrival_delay_mins,
            weather_condition,
            airline_code
        FROM flights_delay
        LIMIT 5
        """
        
        sample = pd.read_sql_query(query, conn)
        print("Sample data from database:")
        print(sample)
        
        conn.close()
        
    except Exception as e:
        print(f"Error checking data: {e}")

# %% [markdown]
# ## Complete!
# The database has been updated with 500 realistic flight delay records.
# You can now run the regression analysis code!