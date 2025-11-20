#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kjmobile/lb/blob/main/1_LM_Simple_Linear_to_Polynomial_Regression_df.ipynb)

# %% [markdown]
# # Simple Linear to Polynomial Regression (DataFrame Ver.)
# This notebook implements linear and polynomial regression using pandas DataFrame directly.

# %% [markdown]
# ## Data Prep

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# %%
# Load data
fish_df = pd.read_csv('https://raw.githubusercontent.com/kjmobile/data/refs/heads/main/ml/fish_l.csv')
fish_df.head()

# %%
# Check data info
fish_df.shape

# %%
fish_df.info()

# %%
# Train-test split (DataFrame version)
train_df, test_df = train_test_split(fish_df, test_size=0.25, random_state=0)

# Separate X and y (keep DataFrame format)
train_X = train_df[['Length']]  # Double brackets to keep DataFrame
train_y = train_df['Weight']
test_X = test_df[['Length']]
test_y = test_df['Weight']

print(f"Train X shape: {train_X.shape}")
print(f"Train y shape: {train_y.shape}")
print(f"Test X shape: {test_X.shape}")
print(f"Test y shape: {test_y.shape}")

# %% [markdown]
# ## Simple Linear Regression
# 
# Model equation: $Weight = \beta_1 \times Length + \beta_0$

# %%
# Train model
m1 = linear_model.LinearRegression()
m1.fit(train_X, train_y)

# %%
# Check coefficients
print(f"Coefficient (β1): {m1.coef_[0]:.2f}")
print(f"Intercept (β0): {m1.intercept_:.2f}")
print(f"\nModel Equation: Weight = {m1.coef_[0]:.2f} × Length + {m1.intercept_:.2f}")

# %% [markdown]
# ### Evaluate Model Performance: m1

# %%
# Evaluate model with R² score
train_r2 = m1.score(train_X, train_y)
test_r2 = m1.score(test_X, test_y)

print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# %%
# Prediction using DataFrame (no warning)
length_60_pred = m1.predict(pd.DataFrame({'Length': [60]}))[0]
print(f"Predicted weight for 60 inch fish: {length_60_pred:.2f} lbs")

# %%
# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(train_X['Length'], train_y, edgecolor='w', alpha=0.7)

# Predictions using DataFrame format
line_x = pd.DataFrame({'Length': [8.4, 60]})
plt.plot([8.4, 60], m1.predict(line_x), ls='--', color='green', linewidth=2, label='Regression Line')

pred_60 = pd.DataFrame({'Length': [60]})
plt.scatter(60, m1.predict(pred_60), color='r', marker="o", s=200, label='60 inch prediction', zorder=5)

plt.xlabel('Length (inch)')
plt.ylabel('Weight (lbs)')
plt.title('Simple Linear Regression')
plt.legend()


# %% [markdown]
# ## Polynomial Regression
# 
# Model equation: $Weight = \beta_2 \times Length^2 + \beta_1 \times Length + \beta_0$

# %%
# Add polynomial features (DataFrame version)
train_X_poly = train_X.copy()
train_X_poly['Length_squared'] = train_X['Length'] ** 2

test_X_poly = test_X.copy()
test_X_poly['Length_squared'] = test_X['Length'] ** 2

# Reorder columns (Length², Length order)
train_X_poly = train_X_poly[['Length_squared', 'Length']]
test_X_poly = test_X_poly[['Length_squared', 'Length']]

print("Train X poly shape:", train_X_poly.shape)
print("\nFirst 5 rows:")
train_X_poly.head()

# %%
# Train model
m2 = linear_model.LinearRegression()
m2.fit(train_X_poly, train_y)

# %%
# Check coefficients
print(f"Coefficient for Length² (β2): {m2.coef_[0]:.2f}")
print(f"Coefficient for Length (β1): {m2.coef_[1]:.2f}")
print(f"Intercept (β0): {m2.intercept_:.2f}")
print(f"\nModel Equation: Weight = {m2.coef_[0]:.2f} × Length² + {m2.coef_[1]:.2f} × Length + {m2.intercept_:.2f}")

# %% [markdown]
# ### Evaluate Model Performance: m2

# %%
# Evaluate model with R² score
train_r2_poly = m2.score(train_X_poly, train_y)
test_r2_poly = m2.score(test_X_poly, test_y)

print(f"Train R²: {train_r2_poly:.4f}")
print(f"Test R²: {test_r2_poly:.4f}")

print("\n=== Model Comparison ===")
print(f"Linear Regression Test R²: {test_r2:.4f}")
print(f"Polynomial Regression Test R²: {test_r2_poly:.4f}")
print(f"Improvement: {(test_r2_poly - test_r2):.4f}")

# %%
# Prediction using DataFrame (no warning)
pred_60_poly = pd.DataFrame({'Length_squared': [60**2], 'Length': [60]})
length_60_pred_poly = m2.predict(pred_60_poly)[0]

print(f"Predicted weight for 60 inch fish (polynomial): {length_60_pred_poly:.2f} lbs")
print(f"Predicted weight for 60 inch fish (linear): {length_60_pred:.2f} lbs")
print(f"Difference: {abs(length_60_pred_poly - length_60_pred):.2f} lbs")

# %%
# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(train_X['Length'], train_y, alpha=0.7)

# Draw polynomial regression line
point = np.arange(8.4, 60, 0.1)
predictions = m2.coef_[0] * point**2 + m2.coef_[1] * point + m2.intercept_
plt.plot(point, predictions, color='blue', linewidth=2, label='Polynomial Regression')

# Prediction for 60 inch using DataFrame
pred_60_poly = pd.DataFrame({'Length_squared': [60**2], 'Length': [60]})
plt.scatter(60, m2.predict(pred_60_poly), color='r', marker="s", s=200, label='60 inch prediction', zorder=5)

plt.xlabel('Length (inch)')
plt.ylabel('Weight (lbs)')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## Compare Both Models

# %%
# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Linear Regression
axes[0].scatter(train_X['Length'], train_y, edgecolor='w', alpha=0.7)
line_x = pd.DataFrame({'Length': [8.4, 60]})
axes[0].plot([8.4, 60], m1.predict(line_x), ls='--', color='green', linewidth=2)
pred_60_linear = pd.DataFrame({'Length': [60]})
axes[0].scatter(60, m1.predict(pred_60_linear), color='r', marker="o", s=200, zorder=5)
axes[0].set_xlabel('Length (inch)')
axes[0].set_ylabel('Weight (lbs)')
axes[0].set_title(f'Linear Regression (R² = {test_r2:.4f})')
axes[0].grid(True, alpha=0.3)

# Polynomial Regression
axes[1].scatter(train_X['Length'], train_y, alpha=0.7)
point = np.arange(8.4, 60, 0.1)
predictions = m2.coef_[0] * point**2 + m2.coef_[1] * point + m2.intercept_
axes[1].plot(point, predictions, color='blue', linewidth=2)
pred_60_poly = pd.DataFrame({'Length_squared': [60**2], 'Length': [60]})
axes[1].scatter(60, m2.predict(pred_60_poly), color='r', marker="s", s=200, zorder=5)
axes[1].set_xlabel('Length (inch)')
axes[1].set_ylabel('Weight (lbs)')
axes[1].set_title(f'Polynomial Regression (R² = {test_r2_poly:.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%

