#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kjmobile/lb/blob/main/5_1_Logistic_Regression_Q2.ipynb)
# 

# # Understanding Logistic Regression
# 
# ## Goal: Predict Flight Cancellations
# 
# In this notebook, we use logistic regression to predict whether a flight will be cancelled.

# ## 1. Load and Explore Data

# In[40]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
url = "https://raw.githubusercontent.com/kjmobile/data/main/ml/flights_50k.csv"
df = pd.read_csv(url)

print(f"Data shape: {df.shape}")
print(f"\nCancellation rate:")
print(df['CANCELLED'].value_counts(normalize=True))


# ## 2. Prepare Data
# 
# Select simple features for logistic regression.
# 
# **Important Notes:**
# - We exclude `DEPARTURE_DELAY` to avoid **data leakage** (we can't know delays before the flight)
# - Cancelled flights have missing `DEPARTURE_DELAY` values (they never departed)
# - This is an **imbalanced dataset** (~1.5% cancellations)

# In[41]:


df.columns


# In[42]:


df.head(2)


# In[43]:


df.columns


# In[44]:


# Select relevant columns (excluding DEPARTURE_DELAY to avoid data leakage)
columns_to_use = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'DISTANCE', 
                  'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'CANCELLED']


df_clean = df[columns_to_use].copy()

# Remove missing values
df_clean = df_clean.dropna()

# Encode AIRLINE as numeric
df_clean['AIRLINE_CODE'] = pd.Categorical(df_clean['AIRLINE']).codes
df_clean['ORIGIN_CODE'] = pd.Categorical(df_clean['ORIGIN_AIRPORT']).codes
df_clean['DEST_CODE'] = pd.Categorical(df_clean['DESTINATION_AIRPORT']).codes
df_clean['DEPARTURE_HOUR'] = df_clean['SCHEDULED_DEPARTURE'] // 100

print(f"Clean data shape: {df_clean.shape}")
print(f"Cancellations: {df_clean['CANCELLED'].sum()} ({df_clean['CANCELLED'].mean():.2%})")
df_clean.head()


# ## 3. Separate Features (X) and Target (y)

# In[45]:


# Select features (no DEPARTURE_DELAY - we can't know this before the flight)
features = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE_CODE', 'DISTANCE', 'ORIGIN_CODE', 'DEST_CODE', 'DEPARTURE_HOUR']
X = df_clean[features]
y = df_clean['CANCELLED']

print(y)
X.head(2)


# ## 4. Split Train/Test Data

# In[46]:


# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")


# ## 5. Train Logistic Regression Model
# 
# Logistic regression is an algorithm for binary classification (0 or 1).
# 
# We use `class_weight='balanced'` to handle the imbalanced dataset by giving more weight to the minority class (cancellations).

# In[47]:


# Create and train model with balanced class weights
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("Model training complete!")


# ## 6. Predict and Evaluate

# In[48]:


# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Cancelled', 'Cancelled']))


# ## 7. Visualize Confusion Matrix

# In[49]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Cancelled', 'Cancelled'],
            yticklabels=['Not Cancelled', 'Cancelled'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# ## 8. Check Feature Importance
# 
# Examine coefficients to see which features are important for predicting cancellations.

# In[50]:


# Feature coefficients
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
})
coefficients = coefficients.sort_values('Coefficient', ascending=False)

print(coefficients)

# Visualize
plt.figure(figsize=(10,3))
plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.xlabel('Coefficient')
plt.title('Logistic Regression Feature Coefficients')
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()


# ## 9. Probability Predictions
# 
# Logistic regression can predict probabilities, not just 0 or 1.

# In[51]:


# Predict probabilities (first 10 samples)
y_prob = model.predict_proba(X_test)[:10]

prob_df = pd.DataFrame({
    'Prob_Not_Cancelled': y_prob[:, 0],
    'Prob_Cancelled': y_prob[:, 1],
    'Predicted': y_pred[:10],
    'Actual': y_test.values[:10]
})

prob_df


# ## 10. Predict New Data Example

# In[52]:


# New flight data (example)
new_flight = pd.DataFrame({
    'MONTH': [12],  # December
    'DAY_OF_WEEK': [1],  # Monday
    'AIRLINE_CODE': [5],
    'DISTANCE': [2500],  # Long distance flight
    'ORIGIN_CODE': [10],  # Example origin airport code
    'DEST_CODE': [25],  # Example destination airport code
    'DEPARTURE_HOUR': [14]  # 2 PM
})

# Predict
prediction = model.predict(new_flight)[0]
probability = model.predict_proba(new_flight)[0]

print(f"Prediction: {'Cancelled' if prediction == 1 else 'Not Cancelled'}")
print(f"Probability of cancellation: {probability[1]:.2%}")
print(f"Probability of not cancelling: {probability[0]:.2%}")


# In[ ]:





# In[ ]:




