#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # Challenge Solutions
# 
# Here are the solutions to the 4 challenges from the regression tutorial.

# %% [markdown]
# ## Challenge 1: Change random_state to 42
# 
# **Question:** Change the `random_state` in train_test_split to 42. Do the R² scores change?
# 
# **Answer:** Yes! The R² scores will change slightly because we're splitting the data differently.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
fish_df = pd.read_csv('https://raw.githubusercontent.com/kjmobile/data/refs/heads/main/ml/fish_l.csv')

# Original split (random_state=0)
train_df_0, test_df_0 = train_test_split(fish_df, test_size=0.25, random_state=0)
train_X_0 = train_df_0[['Length']]
train_y_0 = train_df_0['Weight']
test_X_0 = test_df_0[['Length']]
test_y_0 = test_df_0['Weight']

model_0 = LinearRegression()
model_0.fit(train_X_0, train_y_0)
r2_0 = model_0.score(test_X_0, test_y_0)

# New split (random_state=42)
train_df_42, test_df_42 = train_test_split(fish_df, test_size=0.25, random_state=42)
train_X_42 = train_df_42[['Length']]
train_y_42 = train_df_42['Weight']
test_X_42 = test_df_42[['Length']]
test_y_42 = test_df_42['Weight']

model_42 = LinearRegression()
model_42.fit(train_X_42, train_y_42)
r2_42 = model_42.score(test_X_42, test_y_42)

# Compare
print("=" * 50)
print("CHALLENGE 1: Effect of random_state")
print("=" * 50)
print(f"R² with random_state=0:  {r2_0:.4f}")
print(f"R² with random_state=42: {r2_42:.4f}")
print(f"Difference:              {abs(r2_0 - r2_42):.4f}")
print("\nConclusion: Different splits produce slightly different scores,")
print("but they should be similar if the model is stable.")

# %% [markdown]
# ---
# ## Challenge 2: Create a Cubic Model
# 
# **Question:** Try creating a cubic model by adding a Length³ feature. Does it improve the R² score?
# 
# **Answer:** Let's add Length³ and see!

# %%
# Use original split (random_state=0)
train_df, test_df = train_test_split(fish_df, test_size=0.25, random_state=0)
train_X = train_df[['Length']]
train_y = train_df['Weight']
test_X = test_df[['Length']]
test_y = test_df['Weight']

# Create cubic features: Length³, Length², Length
train_X_cubic = train_X.copy()
train_X_cubic['Length³'] = train_X['Length'] ** 3
train_X_cubic['Length²'] = train_X['Length'] ** 2
train_X_cubic = train_X_cubic[['Length³', 'Length²', 'Length']]

test_X_cubic = test_X.copy()
test_X_cubic['Length³'] = test_X['Length'] ** 3
test_X_cubic['Length²'] = test_X['Length'] ** 2
test_X_cubic = test_X_cubic[['Length³', 'Length²', 'Length']]

print("Cubic features:")
print(train_X_cubic.head())

# Train cubic model
model_cubic = LinearRegression()
model_cubic.fit(train_X_cubic, train_y)

# Compare all three models
model_linear = LinearRegression()
model_linear.fit(train_X, train_y)

train_X_poly = train_X.copy()
train_X_poly['Length²'] = train_X['Length'] ** 2
train_X_poly = train_X_poly[['Length²', 'Length']]
test_X_poly = test_X.copy()
test_X_poly['Length²'] = test_X['Length'] ** 2
test_X_poly = test_X_poly[['Length²', 'Length']]

model_poly = LinearRegression()
model_poly.fit(train_X_poly, train_y)

# Get R² scores
r2_linear = model_linear.score(test_X, test_y)
r2_poly = model_poly.score(test_X_poly, test_y)
r2_cubic = model_cubic.score(test_X_cubic, test_y)

print("\n" + "=" * 50)
print("CHALLENGE 2: Linear vs Polynomial vs Cubic")
print("=" * 50)
print(f"Linear model (degree 1) R²:     {r2_linear:.4f}")
print(f"Polynomial model (degree 2) R²: {r2_poly:.4f}")
print(f"Cubic model (degree 3) R²:      {r2_cubic:.4f}")
print(f"\nImprovement from linear to poly: {(r2_poly - r2_linear):.4f}")
print(f"Improvement from poly to cubic:  {(r2_cubic - r2_poly):.4f}")
print("\nConclusion: The cubic model may improve slightly,")
print("but watch out for overfitting with too many features!")

# Visualize cubic model
plt.figure(figsize=(10, 6))
plt.scatter(train_X['Length'], train_y, alpha=0.6, edgecolor='k', label='Training data')

# Draw cubic curve
lengths = np.linspace(10, 60, 300)
predictions_cubic = (model_cubic.coef_[0] * lengths**3 + 
                     model_cubic.coef_[1] * lengths**2 + 
                     model_cubic.coef_[2] * lengths + 
                     model_cubic.intercept_)
plt.plot(lengths, predictions_cubic, color='purple', linewidth=2, label='Cubic model')

plt.xlabel('Length (inches)')
plt.ylabel('Weight (lbs)')
plt.title(f'Cubic Regression (R² = {r2_cubic:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ---
# ## Challenge 3: Predict Different Lengths
# 
# **Question:** Predict the weight for different lengths (e.g., 30 inches, 45 inches). Which model do you trust more?
# 
# **Answer:** Let's compare predictions at multiple points!

# %%
# Test different lengths
test_lengths = [20, 30, 40, 45, 50, 60]

print("=" * 70)
print("CHALLENGE 3: Predictions at Different Lengths")
print("=" * 70)
print(f"{'Length':<10} {'Linear':<15} {'Polynomial':<15} {'Difference':<15}")
print("-" * 70)

for length in test_lengths:
    # Linear prediction
    pred_linear = model_linear.predict(pd.DataFrame({'Length': [length]}))[0]
    
    # Polynomial prediction
    pred_poly = model_poly.predict(pd.DataFrame({
        'Length²': [length**2], 
        'Length': [length]
    }))[0]
    
    diff = abs(pred_poly - pred_linear)
    
    print(f"{length} inches  {pred_linear:>8.2f} lbs    {pred_poly:>8.2f} lbs    {diff:>8.2f} lbs")

print("\nObservations:")
print("- Predictions are similar near the middle of the data range")
print("- Differences increase at the edges (especially at 60 inches)")
print("- The polynomial model captures the curvature better")
print("\nWhich to trust?")
print("- INSIDE the data range (10-50 inches): Trust polynomial more")
print("- OUTSIDE the data range (>60 inches): Be cautious with both!")

# Visualize predictions
fig, ax = plt.subplots(figsize=(12, 7))

# Plot training data
ax.scatter(train_X['Length'], train_y, alpha=0.6, edgecolor='k', 
          s=50, label='Training data', zorder=3)

# Plot linear model
lengths_plot = np.linspace(8, 65, 300)
preds_linear = model_linear.coef_[0] * lengths_plot + model_linear.intercept_
ax.plot(lengths_plot, preds_linear, color='green', linewidth=2, 
       linestyle='--', label='Linear model')

# Plot polynomial model
preds_poly = model_poly.coef_[0] * lengths_plot**2 + model_poly.coef_[1] * lengths_plot + model_poly.intercept_
ax.plot(lengths_plot, preds_poly, color='blue', linewidth=2, 
       label='Polynomial model')

# Plot test predictions
for length in test_lengths:
    pred_linear = model_linear.predict(pd.DataFrame({'Length': [length]}))[0]
    pred_poly = model_poly.predict(pd.DataFrame({'Length²': [length**2], 'Length': [length]}))[0]
    
    ax.scatter(length, pred_linear, color='green', s=150, marker='o', 
              edgecolor='darkgreen', linewidth=2, zorder=5)
    ax.scatter(length, pred_poly, color='blue', s=150, marker='s', 
              edgecolor='darkblue', linewidth=2, zorder=5)
    
    # Draw line connecting predictions
    ax.plot([length, length], [pred_linear, pred_poly], 
           color='red', linestyle=':', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Length (inches)', fontsize=12)
ax.set_ylabel('Weight (lbs)', fontsize=12)
ax.set_title('Comparing Predictions at Different Lengths', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Challenge 4: Both Models on Same Graph
# 
# **Question:** Plot both the linear and polynomial predictions on the same graph. Where do they differ most?
# 
# **Answer:** Let's create a comprehensive comparison!

# %%
fig, ax = plt.subplots(figsize=(14, 8))

# Plot training data
ax.scatter(train_X['Length'], train_y, alpha=0.6, edgecolor='k', 
          s=80, label='Training data', zorder=3, color='gray')

# Create smooth line for predictions
lengths_smooth = np.linspace(8, 65, 500)

# Linear predictions
preds_linear = model_linear.coef_[0] * lengths_smooth + model_linear.intercept_
ax.plot(lengths_smooth, preds_linear, color='green', linewidth=3, 
       linestyle='--', label=f'Linear model (R²={r2_linear:.4f})', alpha=0.8)

# Polynomial predictions
preds_poly = (model_poly.coef_[0] * lengths_smooth**2 + 
              model_poly.coef_[1] * lengths_smooth + 
              model_poly.intercept_)
ax.plot(lengths_smooth, preds_poly, color='blue', linewidth=3, 
       label=f'Polynomial model (R²={r2_poly:.4f})', alpha=0.8)

# Highlight difference area
ax.fill_between(lengths_smooth, preds_linear, preds_poly, 
               alpha=0.2, color='red', label='Prediction difference')

# Mark key points
special_lengths = [15, 30, 45, 60]
for length in special_lengths:
    pred_linear = model_linear.coef_[0] * length + model_linear.intercept_
    pred_poly = model_poly.coef_[0] * length**2 + model_poly.coef_[1] * length + model_poly.intercept_
    diff = abs(pred_poly - pred_linear)
    
    # Add annotation
    ax.annotate(f'{length}" fish\nDiff: {diff:.1f} lbs', 
               xy=(length, pred_poly), 
               xytext=(length + 3, pred_poly + 20),
               fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))

ax.set_xlabel('Length (inches)', fontsize=13, fontweight='bold')
ax.set_ylabel('Weight (lbs)', fontsize=13, fontweight='bold')
ax.set_title('Linear vs Polynomial: Where Do They Differ?', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(8, 65)

plt.tight_layout()
plt.show()

# Calculate and display maximum difference
differences = np.abs(preds_poly - preds_linear)
max_diff_idx = np.argmax(differences)
max_diff_length = lengths_smooth[max_diff_idx]
max_diff_value = differences[max_diff_idx]

print("\n" + "=" * 60)
print("CHALLENGE 4: Where Models Differ Most")
print("=" * 60)
print(f"Maximum difference occurs at: {max_diff_length:.1f} inches")
print(f"Difference at that point:     {max_diff_value:.2f} lbs")
print("\nKey Observations:")
print("1. Small differences in the middle range (20-40 inches)")
print("2. Differences grow at the edges (especially beyond 50 inches)")
print("3. The polynomial curve bends, while linear stays straight")
print("4. For very long fish (>55 inches), predictions diverge significantly")
print("\nConclusion:")
print("- Use polynomial model for predictions within the data range")
print("- Be very careful with extrapolation beyond the training data!")

# %%
