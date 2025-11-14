#!/usr/bin/env python
# coding: utf-8

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kjmobile/lb/blob/main/1_LM_Simple_Linear_to_Polynomial_Regression_ndarray.ipynb)

# # From Simple Linear to Polynomial Regression (NumPy Array Version)
# 
# Now that you've learned regression with pandas DataFrames, let's do the same thing using **NumPy arrays**.
# 
# **Why learn the array version?**
# - Many machine learning libraries work directly with arrays
# - Arrays are faster for numerical computations
# - Understanding arrays helps you work with any ML framework
# 
# **What's different?**
# - We'll use `.values` to convert DataFrames to arrays
# - Array indexing works differently: `X[:, 0]` instead of `X['Length']`
# - We need to reshape 1D arrays to 2D for sklearn: `.reshape(-1, 1)`
# 

# ## 1. Import Libraries
# 
# Same libraries as before!
# 

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ## 2. Load Data and Convert to Arrays
# 
# We'll start with a DataFrame, then convert to NumPy arrays.
# 

# In[33]:


# Load data as DataFrame
fish_df = pd.read_csv('https://raw.githubusercontent.com/kjmobile/data/refs/heads/main/ml/fish_l.csv')
fish_df.head()


# In[34]:


# Convert to NumPy arrays
fish_length = fish_df['Length'].values  # 1D array
fish_weight = fish_df['Weight'].values  # 1D array


# In[35]:


print(f"Length: {fish_length.shape}")  # (159,) - 1D array with 159 elements
print(f"Weight: {fish_weight.shape}")  # (159,) - 1D array with 159 elements


# **Understanding array shapes:**
# - `(159,)` means a 1D array with 159 elements
# - `(159, 1)` means a 2D array with 159 rows and 1 column
# - sklearn needs 2D arrays for features (X), but 1D is OK for targets (y)
# 

# ### Split into Training and Test Sets
# 
# When splitting arrays, we get arrays back (not DataFrames).
# 

# In[36]:


# For sklearn, we need X to be 2D
X = fish_length.reshape(-1, 1)  # Convert 1D to 2D: (159,) -> (159, 1)
y = fish_weight                  # Keep y as 1D: (159,)

print(f"X shape after reshape: {X.shape}")  # (159, 1) - 2D array
print(f"y shape: {y.shape}")                 # (159,) - 1D array


# In[40]:


# Split the data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=0)

print(f"\nTraining set: {len(train_X)} fish")
print(f"Test set: {len(test_X)} fish")


# **Why `.reshape(-1, 1)`?**
# - `-1` means "figure out this dimension automatically"
# - `1` means we want 1 column
# - So (159,) becomes (159, 1)
# 
# **Alternative ways to reshape:**
# ```python
# X = fish_length.reshape(159, 1)  # Explicit
# X = fish_length[:, np.newaxis]   # Add new axis
# X = fish_length[:, None]         # Same as above
# ```
# 

# ---
# ## 3. Simple Linear Regression with Arrays
# 
# The model training is identical! sklearn works the same with arrays.
# 

# In[43]:


# Create and train the model
model_linear = LinearRegression()
model_linear.fit(train_X, train_y)

# Check what the model learned
print(f"Slope (β₁): {model_linear.coef_[0]:.2f}")
print(f"Intercept (β₀): {model_linear.intercept_:.2f}")


# ### Evaluate the Model
# 

# In[44]:


train_r2 = model_linear.score(train_X, train_y)
test_r2 = model_linear.score(test_X, test_y)

print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")


# ### Make a Prediction with Arrays
# 
# **Key difference:** We need to create a 2D array for prediction!
# 

# In[42]:


# Predict for a 60-inch fish
length_60 = np.array([[60]])  # 2D array: shape (1, 1)
predicted_weight = model_linear.predict(length_60)[0]

print(f"Input shape: {length_60.shape}")  # (1, 1)
print(f"A 60-inch fish should weigh approximately {predicted_weight:.2f} lbs")


# **Different ways to create the input:**
# ```python
# length_60 = np.array([[60]])           # Method 1: nested list
# length_60 = np.array([60]).reshape(1, 1)  # Method 2: reshape
# length_60 = np.array([60]).reshape(-1, 1) # Method 3: auto dimension
# ```
# 

# ### Visualize the Model
# 

# In[72]:


plt.figure(figsize=(10, 6))
plt.scatter(train_X, train_y, alpha=0.6, edgecolor='k', label='Training data')

# Draw the regression line
line_X = np.array([[10], [60]])  # 2D array with 2 points
plt.plot(line_X, model_linear.predict(line_X), 
         color='green', linewidth=2, linestyle='--', label='Linear model')

plt.scatter(60, predicted_weight, color='red', s=200, marker='o', 
           edgecolor='k', label='Prediction for 60"', zorder=5)

plt.xlabel('Length (inches)')
plt.ylabel('Weight (lbs)')
plt.title('Simple Linear Regression (Array Version)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ---
# ## 4. Polynomial Regression with Arrays
# 
# Here's where arrays get interesting! We need to manually create polynomial features.
# 

# ### Create Polynomial Features
# 
# **With DataFrames:** We added a new column  
# **With arrays:** We use `np.column_stack()` or `np.hstack()` to combine arrays
# 

# In[79]:


a=train_X**2
b=test_X**2


# In[94]:


np.column_stack([a, train_X]).shape


# In[97]:


np.vstack([a, train_X]).shape


# In[101]:


# Create Length² feature
train_X_squared = train_X ** 2  # Element-wise squaring
test_X_squared = test_X ** 2

print(f"Original train_X shape: {train_X.shape}")      # (119, 1)
print(f"Squared train_X shape: {train_X_squared.shape}")  # (119, 1)

# Combine Length² and Length into one array
train_X_poly = np.column_stack([train_X_squared, train_X])
test_X_poly = np.column_stack([test_X_squared, test_X])

print(f"\nPolynomial features shape: {train_X_poly.shape}")  # (119, 2)
print("\nFirst 5 rows of polynomial features:")
print(train_X_poly[:5])  # Show first 5 rows


# **Understanding `np.column_stack()`:**
# - Takes a list of arrays: `[array1, array2]`
# - Stacks them side-by-side as columns
# - Result: a 2D array with multiple columns
# 
# **Alternative method using `np.hstack()`:**
# ```python
# train_X_poly = np.hstack([train_X_squared, train_X])
# ```
# 
# **What does the array look like?**
# - Column 0: Length² values
# - Column 1: Length values
# - Each row: one fish
# 

# ### Train the Polynomial Model
# 

# In[104]:


model_poly = LinearRegression()
model_poly.fit(train_X_poly, train_y)

print(f"Coefficient for Length² (β₂): {model_poly.coef_[0]:.4f}")
print(f"Coefficient for Length (β₁): {model_poly.coef_[1]:.2f}")
print(f"Intercept (β₀): {model_poly.intercept_:.2f}")


# ### Evaluate the Polynomial Model
# 

# In[105]:


train_r2_poly = model_poly.score(train_X_poly, train_y)
test_r2_poly = model_poly.score(test_X_poly, test_y)

print(f"Training R²: {train_r2_poly:.4f}")
print(f"Test R²: {test_r2_poly:.4f}")
print(f"\n{'='*40}")
print("COMPARISON:")
print(f"{'='*40}")
print(f"Linear model test R²:     {test_r2:.4f}")
print(f"Polynomial model test R²: {test_r2_poly:.4f}")
print(f"Improvement:              {(test_r2_poly - test_r2):.4f}")


# ### Make a Prediction with Arrays
# 
# For a 60-inch fish, we need BOTH Length² and Length!
# 
# For a 60-inch fish, we need BOTH Length<sup>2</sup> and Length!

# In[106]:


# Create input: [Length², Length]
length_60_poly = np.array([[60**2, 60]])  # Shape: (1, 2)
predicted_weight_poly = model_poly.predict(length_60_poly)[0]

print(f"Input array: {length_60_poly}")
print(f"Input shape: {length_60_poly.shape}")  # (1, 2)
print(f"\nPolynomial model predicts: {predicted_weight_poly:.2f} lbs")
print(f"Linear model predicts:     {predicted_weight:.2f} lbs")
print(f"Difference:                {abs(predicted_weight_poly - predicted_weight):.2f} lbs")


# **Breaking down the input:**
# ```python
# length_60_poly = np.array([[60**2, 60]])
#                            ↑       ↑
#                         Length²  Length
# ```
# - `60**2 = 3600` (Length²)
# - `60` (Length)
# - Shape must be (1, 2) because we trained on 2 features
# 

# ### Visualize the Polynomial Model
# 

# In[108]:


plt.figure(figsize=(6, 6))
plt.scatter(train_X, train_y, alpha=0.6, edgecolor='k', label='Training data')

# Draw the curved line
lengths = np.linspace(10, 60, 300)  # 300 points between 10 and 60
# Calculate predictions manually using the equation
predictions = model_poly.coef_[0] * lengths**2 + model_poly.coef_[1] * lengths + model_poly.intercept_
plt.plot(lengths, predictions, color='blue', linewidth=2, label='Polynomial model')

plt.scatter(60, predicted_weight_poly, color='red', s=200, marker='s', 
           edgecolor='k', label='Prediction for 60"', zorder=5)

plt.xlabel('Length (inches)')
plt.ylabel('Weight (lbs)')
plt.title('Polynomial Regression (Array Version)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ---
# ## 5. Side-by-Side Comparison
# 

# In[25]:


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Linear Regression
axes[0].scatter(train_X, train_y, alpha=0.6, edgecolor='k')
line_X = np.array([[10], [60]])
axes[0].plot(line_X, model_linear.predict(line_X), 
            color='green', linewidth=2, linestyle='--')
axes[0].scatter(60, predicted_weight, color='red', s=200, marker='o', edgecolor='k', zorder=5)
axes[0].set_xlabel('Length (inches)', fontsize=12)
axes[0].set_ylabel('Weight (lbs)', fontsize=12)
axes[0].set_title(f'Linear Model (R² = {test_r2:.4f})', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Polynomial Regression
axes[1].scatter(train_X, train_y, alpha=0.6, edgecolor='k')
lengths = np.linspace(10, 60, 300)
predictions = model_poly.coef_[0] * lengths**2 + model_poly.coef_[1] * lengths + model_poly.intercept_
axes[1].plot(lengths, predictions, color='blue', linewidth=2)
axes[1].scatter(60, predicted_weight_poly, color='red', s=200, marker='s', edgecolor='k', zorder=5)
axes[1].set_xlabel('Length (inches)', fontsize=12)
axes[1].set_ylabel('Weight (lbs)', fontsize=12)
axes[1].set_title(f'Polynomial Model (R² = {test_r2_poly:.4f})', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ---
# ## 6. Key Differences: DataFrame vs Array
# 
# | Aspect | DataFrame | NumPy Array |
# |--------|-----------|-------------|
# | **Column access** | `df['Length']` | `X[:, 0]` |
# | **Shape for sklearn** | Already 2D | Need `.reshape(-1, 1)` |
# | **Add features** | `df['Length²'] = df['Length']**2` | `np.column_stack([X², X])` |
# | **Prediction input** | `pd.DataFrame({'Length': [60]})` | `np.array([[60]])` |
# | **Column names** | Yes, easy to read | No, must track manually |
# | **Speed** | Slower for computation | Faster for computation |
# 
# **When to use which?**
# - **DataFrames**: Data exploration, readability, mixed data types
# - **Arrays**: Production code, speed, pure numerical computation
# 

# ---
# ## Try It Yourself!
# 
# **Challenge 1:** Change the `random_state` to 42 and see if the R² scores change.
# 
# **Challenge 2:** Create a cubic model using arrays. You'll need to stack three arrays: [Length³, Length², Length]
# 
# **Challenge 3:** Explain this array indexing code:
# ```python
# first_fish = train_X[0, 0]      # What does this get?
# first_10_fish = train_X[:10, 0]  # What about this?
# all_lengths = train_X[:, 0]      # And this?
# ```
# 

# ---
# 

# #### Challenge 1: Change random_state to 42
# 

# In[109]:


# Your code here



# #### Challenge 2: Create a Cubic Model with Arrays
# 
# **Hint:** You'll need to create three arrays and stack them:
# ```python
# train_X_cubic = np.column_stack([train_X**3, train_X**2, train_X])
# ```
# 

# In[110]:


# Your code here
# Challenge 2: Cubic model with arrays, extend the following code
train_X_cubic = np.column_stack([train_X**3, train_X**2, train_X])
test_X_cubic = np.column_stack([test_X**3, test_X**2, test_X])



model_cubic = LinearRegression()
model_cubic.fit(train_X_cubic, train_y)
print(f"Cubic R²: {model_cubic.score(test_X_cubic, test_y):.4f}")




# #### Challenge 3: Array Indexing Explanation
# 
# Explain what each line does:
# ```python
# first_fish = train_X[0, 0]      # ?
# first_10_fish = train_X[:10, 0] # ?
# all_lengths = train_X[:, 0]     # ?
# ```
# 
Your answer here:



# In[114]:


# print the three variables created above
first_fish = train_X[0, 0]       # Gets the first row, first column (one value)
first_10_fish = train_X[:10, 0]  # Gets first 10 rows, first column (10 values)  
all_lengths = train_X[:, 0]      # Gets all rows, first column (all values)

print(first_10_fish)


# ---
# ## Summary
# 
# 1. **Converting DataFrames to arrays** using `.values`
# 2. **Reshaping arrays** with `.reshape(-1, 1)` for sklearn
# 3. **Creating polynomial features** with `np.column_stack()`
# 4. **Array indexing** with `[row, column]` notation
# 5. **Why arrays matter** - they're fundamental to ML
# 
# 
# 

# In[ ]:




