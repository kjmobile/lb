#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kjmobile/lb/blob/main/6_Decision_Tree_Q.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Decision Tree vs. Logistic Regression

# ## Logistic regression

# In[2]:


import pandas as pd
wine = pd.read_csv('https://raw.githubusercontent.com/kjmobile/data/main/ml/wine_csv.csv')


# In[3]:


wine.shape


# In[4]:


wine.head()


# In[50]:


wine['class'].nunique()


# In[5]:


# prints information about a DataFrame
wine.info()


# In[6]:


wine.describe()


# In[7]:


data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()


# In[9]:


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=17)


# In[10]:


print(train_input.shape, test_input.shape)


# In[11]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# In[12]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))


# In[13]:


get_ipython().run_line_magic('pinfo', 'lr.score')


# ### Interpretability of models (logistic regression vs. decision tree)

# In[14]:


print(lr.coef_, lr.intercept_)


# ## Decision Tree

# In[15]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy', random_state=17) # what's the default criterion in sklearn?
dt.fit(train_scaled, train_target) # what is target variable here?

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target)) # does it show the model is overfitted?


# In[26]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alchol','sugar','ph'])
plt.show()


# In[38]:


get_ipython().run_line_magic('pinfo', 'plot_tree')


# ### Pruning

# In[35]:


# prune to avoid overfitting.
dt = DecisionTreeClassifier(max_depth=4, random_state=17, criterion='log_loss')
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target)) # did this ameliorate overfitting ?


# In[37]:


# Why chose max_depth= 4? : Draw a comparison of train vs test scores through a plot by changing max_depth from 3 to 10

train_scores = []
test_scores = []

for max_depth in range(3, 11):
    dt1 = DecisionTreeClassifier(max_depth=max_depth, random_state=17, criterion='entropy')
    dt1.fit(train_scaled, train_target)

    train_scores.append(dt1.score(train_scaled, train_target))
    test_scores.append(dt1.score(test_scaled, test_target))

plt.figure(figsize=(10, 7))
plt.plot(range(3, 11), train_scores, label="train")
plt.plot(range(3, 11), test_scores, label="test")
plt.xlabel("max_depth")
plt.ylabel("score")
plt.legend()
plt.show()


# In[38]:


# now plot the tree!
plt.figure(figsize=(12,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH']) # when filled=True, does the filled color have any meaning ?
plt.show()


# In[39]:


print(dt.feature_importances_) # what does the feature_importance_ show and how, it is used in other modeling?


# In[56]:


plt.barh(range(3), width=dt.feature_importances_, label=['alchol', 'sugar', 'ph'])
plt.legend()


# Feature importance represents the contribution ratio of each feature to the impurity reduction in a decision tree. The calculation formula is as follows:
# 
# 
# Feature Importance =
# $$\frac{\text{Total Impurity Reduction by the Feature}}{\text{Sum of Impurity Reductions Across All Features}}$$
# 

# ## More to understand

# ### Predict categories using the test_input
# 

# In[57]:


dt_prediction = dt.predict(test_input)
print(dt_prediction)


# ### Show Confusion Matrix

# In[58]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_target, dt_prediction)
cm_df=pd.DataFrame(cm, columns=['predicted_0', 'predict_1'], index=['actual_0','actual_1'])
print(cm_df)


# In[59]:


plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


# ### Information Gain

# In[60]:


plt.figure(figsize=(5,4))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH']) # what if you do not pass the argument of feature_names?
plt.show()


# In[61]:


# In the above chart the inital entropy is .806 and
# when the data (5197) is split ( 2954 [on left] + 2243 [on right]), entropy per each is 0.975 and 0.22
# Hence the information gain by this split is:

initial_entropy = 0.806
left_entropy = 0.975
right_entropy = 0.22
left_proportion = 2954 / (2954 + 2243)
right_proportion = 2243 / (2954 + 2243)
entropy_after_split = left_proportion * left_entropy + right_proportion * right_entropy
information_gain = 0  # fix the line here to compute information gain correctly.
print(f'I(Dp) : {initial_entropy}')
print(f'I(Dj): {entropy_after_split}')
print(f'IG : {information_gain}')


# In[62]:


initial_entropy - entropy_after_split


# ### Using min_impurity_decrease instead of max_depth

# In[63]:


dt2 = DecisionTreeClassifier(min_impurity_decrease=0.001, random_state=17) #min_impurity_decrease= is used to replace what parameter used above?
dt2.fit(train_input, train_target)

print(dt2.score(train_input, train_target))
print(dt2.score(test_input, test_target))


# ## Entropy vs. Gini Impurity

# $$Entropy(D) = -\sum_{i=1}^{n}p_ilog_2(p_i)$$
# $$Gini(D) = \sum_{i=1}^{n}p_i^2$$

# Gini impurity and entropy are in a monotonic relationship, meaning they increase or decrease together based on the uniformity or imbalance of the probability distribution.
# Gini impurity is more commonly used in decision trees than entropy because it has a lower computational cost.

# In[ ]:





# In[64]:


from sympy import *
import matplotlib.pyplot as plt

a = symbols('a')
b = 1.0 - a
gini_impurity_f = 1 - a**2 - b**2
epsilon = 1e-10
entropy_f = - (a + epsilon) * log(a + epsilon, 2) - (b + epsilon) * log(b + epsilon, 2)

# Create the plots individually
p1 = plot(gini_impurity_f, (a, 0, 1), label='Gini', show=False,line_color='red')  # Set show=False to prevent display
p2 = plot(entropy_f, (a, 0, 1), label='Entropy', show=False, line_color='blue')

# Combine the plots
p1.append(p2[0])  # Append the second plot's line object to the first plot
p1.title =" Entroy vs. Gini when n = 2"
p1.legend= True
p1.show()  # Show the combined plot


# In[ ]:




