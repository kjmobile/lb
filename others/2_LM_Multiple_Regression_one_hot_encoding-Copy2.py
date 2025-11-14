#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kjmobile/lb/blob/main/2_LM_Multiple_Regression_one_hot_encoding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## 1 Library and Data :

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import Image
from sklearn.linear_model import LinearRegression


# In[69]:


startup= pd.read_csv("https://raw.githubusercontent.com/kjmobile/data/refs/heads/main/ml/startup.csv", index_col=0)
startup.head()


# In[70]:


from matplotlib import pyplot as plt
startup.plot(kind='scatter', x='Marketing Spend', y='Profit', s=startup['R&D Spend']/1000, alpha=.8)


# ## 2 Check Data

# In[71]:


startup.describe()


# In[72]:


startup.info()


# ## 3 Preprocessing :
#  - 1) one-hot encoding (cf. dummy variable)  
#  - 2) train/test split

# In[73]:


startup.head(3)


# In[74]:


# One-hot encode on state column and save the data as startup_encoded
import pandas as pd
startup_encoded = pd.get_dummies(startup, columns=['State'])


# In[75]:


startup_encoded.head(2)


# In[76]:


startup_encoded.info()


# In[77]:


startup_encoded.columns


# In[78]:


startup_encoded.head(3)


# In[79]:


startup_encoded.columns !='Profit'


# In[80]:


# what does this code do?
X=startup_encoded.loc[:, startup_encoded.columns!='Profit']
y=startup_encoded['Profit']


# In[81]:


X.head(2)


# In[82]:


y.head(2)


# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


# how much percent of data is being using for training?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# ## 4 Modeling

# In[85]:


from sklearn.linear_model import LinearRegression


# In[86]:


model = LinearRegression()


# In[87]:


model.fit(X_train, y_train)


# In[88]:


model_1 = LinearRegression(fit_intercept=False)


# In[89]:


model_1.fit(X_train, y_train)


# Note: In this example, we took only machine learning approach aiming for predictive performance.
# For interpretability and coefficient estimation in linear models, we must omit one dummy to avoid multicorelinarity;
# For predictive performance in models, including all dummies, as we do here, might be acceptable.

# ## 5 Prediction

# In[90]:


pred = model.predict(X_test)
pred


# In[91]:


pred_1= model_1.predict(X_test)
pred_1


# ## 6 Evaluating the model

# In[92]:


comparison = pd.DataFrame({'actual': y_test, 'pred': pred})


# In[93]:


comparison_1 = pd.DataFrame({'actual': y_test, 'pred': pred_1})


# In[94]:


comparison_1


# In[95]:


comparison


# In[96]:


# Compare actual v. predicted with line vs. scatter comparsion

# Plot the actual and predicted values
plt.scatter(y_test, pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted Values")

# Add a line for the perfect fit
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red")

# Show the plot
plt.show()


# In[45]:


from sklearn.metrics import mean_squared_error


# In[46]:


mse=mean_squared_error(y_test, pred)  #mean squared error
mse


# In[59]:


mse_1=mean_squared_error(y_test, pred_1)  #mean squared error
mse_1


# In[47]:


rmse= mse ** 0.5   #root mean squared error
rmse


# In[60]:


rmse_1= mse_1 ** 0.5   #root mean squared error
rmse_1


# In[48]:


model.score(X_train, y_train) # r-squared on train set


# In[61]:


model_1.score(X_train, y_train) # r-squared on train set


# In[49]:


model.score(X_test, y_test) # r-squared on test set  : 1) which r-squared is more important? 2) are they the same as statmodels' r-squared?


# In[62]:


model_1.score(X_test, y_test) # r-squared on test set  : 1) which r-squared is more important? 2) are they the same as statmodels' r-squared?


# In[64]:


model_1.coef_


# In[65]:


model.coef_


# In[68]:


X_test.columns


# ---
