#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kjmobile/lb/blob/main/2_LM_Multiple_Regression_one_hot_encoding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## 1 Library and Data :

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import Image
from sklearn.linear_model import LinearRegression


# In[2]:


startup= pd.read_csv("https://raw.githubusercontent.com/kjmobile/data/refs/heads/main/ml/startup.csv", index_col=0)
startup.head()


# In[3]:


from matplotlib import pyplot as plt
startup.plot(kind='scatter', x='Marketing Spend', y='Profit', s=32, alpha=.8)


# ## 2 Check Data

# In[4]:


startup.describe()


# In[5]:


startup.info()


# ## 3 Preprocessing :
#  - 1) one-hot encoding (cf. dummy variable)  
#  - 2) train/test split

# In[6]:


# One-hot encode on state column and save the data as startup_encoded
import pandas as pd
startup_encoded = pd.get_dummies(startup, columns=['State'])


# In[25]:


# One-hot encode on state column and save the data as startup_encoded
import pandas as pd
startup_encoded_dum = pd.get_dummies(startup, columns=['State'],drop_first=True)


# In[7]:


startup_encoded.info()


# In[8]:


startup_encoded.columns


# In[26]:


startup_encoded_dum.columns


# In[9]:


startup_encoded.head(3)


# In[10]:


startup_encoded.columns !='Profit'


# In[27]:


startup_encoded_dum.columns !='Profit'


# In[29]:


# what does this code do?
X_dum=startup_encoded_dum.loc[:, startup_encoded_dum.columns!='Profit']
y=startup_encoded_dum['Profit']


# In[11]:


# what does this code do?
X=startup_encoded.loc[:, startup_encoded.columns!='Profit']
y=startup_encoded['Profit']


# In[12]:


X.head(2)


# In[13]:


y.head(2)


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


# how much percent of data is being using for training?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# In[30]:


# how much percent of data is being using for training?
X_train_dum, X_test_dum, y_train, y_test = train_test_split(X_dum, y, test_size = 0.2, random_state=0)


# ## 4 Modeling

# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


model = LinearRegression()


# In[19]:


model_1=LinearRegression(fit_intercept=False)


# In[19]:


model_1=LinearRegression()


# In[31]:


model_dum=LinearRegression()


# In[20]:


model.fit(X_train, y_train)


# In[34]:


model_dum.fit(X_train_dum, y_train)


# In[21]:


model_1.fit(X_train, y_train)


# In[35]:


model_dum.coef_


# In[23]:


model.coef_


# In[22]:


model_1.coef_


# Note: In this example, we took only machine learning approach aiming for predictive performance.
# For interpretability and coefficient estimation in linear models, we must omit one dummy to avoid multicorelinarity;
# For predictive performance in models, including all dummies, as we do here, might be acceptable.

# ## 5 Prediction

# In[21]:


pred = model.predict(X_test)
pred


# ## 6 Evaluating the model

# In[22]:


comparison = pd.DataFrame({'actual': y_test, 'pred': pred})


# In[23]:


comparison


# In[24]:


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


# In[25]:


from sklearn.metrics import mean_squared_error


# In[26]:


mse=mean_squared_error(y_test, pred)  #mean squared error
mse


# In[27]:


rmse= mse ** 0.5   #root mean squared error
rmse


# In[28]:


model.score(X_train, y_train) # r-squared on train set


# In[29]:


model.score(X_test, y_test) # r-squared on test set  : 1) which r-squared is more important? 2) are they the same as statmodels' r-squared?


# ---

# # Challenge: 

# >Create a new model named model_1 that avoids (or solves) the multicollinearity problem by setting the fit_intercept parameter to False.
# >After fitting model_1, compare its coefficients (model_1.coef_) with the original model.coef_.
# 
# - Identify which coefficients (especially the State variables) have changed significantly.
# - Explain why this change happened. (Hint: Think about what the coefficients represent when there is no intercept vs. when there is one.)

# In[ ]:




