#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kjmobile/lb/blob/main/5_Logistic_Regression_Q.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## 1 Libraries
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -  1.1. Logistic Function (aka, Sigmoid Function) :  
# $$P(y=1|X) = \frac{1}{1+e^{-z}}$$
# 
# 
# $$\text where $$
# 
# $$ z = \beta_0 + \beta_1X$$

# In[9]:


z = np.linspace(-5,5,100)
plt.plot(z)


# $z = \beta_1 x + \beta_0$

# In[43]:


plt.scatter(range(0,20), np.random.randn(20)*3+range(0,20))


# In[38]:


plt.scatter(range(0,7),[0]*3+[1]+[1]*3)


# $z= \beta_1 x + \beta_0 $

# In[51]:


plt.plot(1/(1+np.exp(-z)))
plt.xlabel("z=ax+b")
plt.ylabel('y');

p = 1/(1+exp(-z))
1/p=1+exp(-z)
1/p=1+exp(-z)
(1/p)-1=exp(-z)
(1-p)/p=exp(-z)
p/(1-p) =exp(z)
ln(p/1-p)=z = ax+b
ln(p/1-p)=ax+b
# In[53]:


np.exp(0.5)


# ## 2 Import Dataset, titanic

# In[54]:


# prompt: get titanic dataset from sns
data=pd.read_csv("https://raw.githubusercontent.com/kjmobile/data/main/ml/titanic.csv")


# In[55]:


data.head(2)


# ## 3 Checking titanic

# In[56]:


data.info()


# In[57]:


data.corr(numeric_only=True)


# In[68]:


sns.heatmap(data.corr(numeric_only=True), cmap='Greys', linecolor='w', annot=True, cbar=0)


# ## 4 Preprocessing: Converting Categorical Variables (Dummy Variables and One-Hot Encoding)

# In[69]:


data.head(2)


# In[70]:


titanic = data.drop(['Name','Ticket'], axis=1)
titanic.head()


# In[ ]:


# We use dummy coding (instead of one hot encoding this time by setting drop_first=True)
titanic_dum=pd.get_dummies(titanic, columns = ['Sex','Embarked'], drop_first=True)


# In[ ]:


titanic_dum


# ## 5 Modeling and Predicting

# In[ ]:


from sklearn.model_selection import train_test_split

X = titanic_dum.drop('Survived', axis = 1)
y = titanic_dum['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


get_ipython().run_line_magic('pinfo', 'lr.fit')


# In[ ]:


X_train.head(2)


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[ ]:


# Predict category
pred = lr.predict(X_test)


# In[ ]:


# to obtain actual probablities
lr.predict_proba(X_test)


# ## 6 Evaluating Prediction Models

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)


# In[ ]:


pd.Series(lr.coef_[0], index = X.columns)


# ## 7 Apply some Feature Engineering

# In[ ]:


titanic_dum.columns


# In[ ]:


titanic_dum['family'] = titanic_dum['SibSp'] + titanic_dum['Parch'] # sibling & spouse + Parent & child


# In[ ]:


titanic_dum.head()


# In[ ]:


# Dose the prediction accuracy improved by 'engineering' "family variable?"
X = titanic_dum.drop(['Survived','SibSp','Parch'], axis = 1)
y = titanic_dum['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
accuracy_score(y_test, pred)


# In[ ]:


# visualize the predicted performance: pink (false positive) and black (fasle nagative) bars represents incorrect prediction

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 1))

plt.bar(range(len(y_test)), y_test+1, label='actual+1', color='black')
plt.bar(range(len(pred)), (pred+1), label='pred+1', color='red', alpha=0.5)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of pred vs actual (y_test)')
plt.legend(ncol=3, loc=(0,1.01))
plt.show()

