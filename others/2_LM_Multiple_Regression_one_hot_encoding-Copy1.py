#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kjmobile/lb/blob/main/2_LM_Multiple_Regression_one_hot_encoding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## 1 Library and Data :

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import Image
from sklearn.linear_model import LinearRegression


# In[5]:


startup= pd.read_csv("https://raw.githubusercontent.com/kjmobile/data/refs/heads/main/ml/startup.csv")
startup.head()


# In[6]:


from matplotlib import pyplot as plt
startup.plot(kind='scatter', x='Marketing Spend', y='Profit', s=32, alpha=.8)


# ## 2 Check Data

# In[7]:


startup.describe()


# In[8]:


startup.info()


# ## 3 Preprocessing :
#  - 1) one-hot encoding (cf. dummy variable)  
#  - 2) train/test split

# In[9]:


startup['State'].unique()


# In[10]:


startup.head(3)


# In[11]:


pd.get_dummies(startup, columns=['State'])


# In[43]:


# One-hot encode on state column and save the data as startup_encoded
import pandas as pd
startup_encoded = pd.get_dummies(startup, columns=['State'])


# In[44]:


startup_encoded.info()


# In[45]:


startup_encoded.columns


# In[46]:


startup_encoded.head(3)


# In[47]:


startup_encoded.columns !='Profit'


# In[48]:


# what does this code do?
X=startup_encoded.loc[:, startup_encoded.columns!='Profit']
y=startup_encoded['Profit']


# In[49]:


X.head(2)


# In[13]:


y.head(2)


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


# how much percent of data is being using for training?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# ## 4 Modeling

# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


model = LinearRegression()


# In[18]:


model.fit(X_train, y_train)


# Note: In this example, we took only machine learning approach aiming for predictive performance.
# For interpretability and coefficient estimation in linear models, we must omit one dummy to avoid multicorelinarity;
# For predictive performance in models, including all dummies, as we do here, might be acceptable.

# ## 5 Prediction

# In[19]:


pred = model.predict(X_test)
pred


# ## 6 Evaluating the model

# In[20]:


comparison = pd.DataFrame({'actual': y_test, 'pred': pred})


# In[21]:


comparison


# In[22]:


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


# In[23]:


from sklearn.metrics import mean_squared_error


# In[24]:


mse=mean_squared_error(y_test, pred)  #mean squared error
mse


# In[25]:


rmse= mse ** 0.5   #root mean squared error
rmse


# In[26]:


model.score(X_train, y_train) # r-squared on train set


# In[27]:


model.score(X_test, y_test) # r-squared on test set  : 1) which r-squared is more important? 2) are they the same as statmodels' r-squared?


# ---
