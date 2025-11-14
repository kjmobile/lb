#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/kjmobile/lb/blob/main/6_Decision_Tree_Q.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Decision Tree vs. Logistic Regression

# ## Logistic regression

# In[1]:


import pandas as pd
wine = pd.read_csv('https://raw.githubusercontent.com/kjmobile/data/main/ml/wine_csv.csv')


# In[2]:


wine.shape


# In[3]:


wine.head()


# In[4]:


wine['class'].nunique()


# In[5]:


# prints information about a DataFrame
wine.info()


# In[6]:


wine.describe()


# In[7]:


data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()


# In[8]:


from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=17)


# In[9]:


print(train_input.shape, test_input.shape)


# In[10]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# In[11]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))


# In[12]:


get_ipython().run_line_magic('pinfo', 'lr.score')


# ### Interpretability of models (logistic regression vs. decision tree)

# In[13]:


print(lr.coef_, lr.intercept_)


# ## Decision Tree

# In[14]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy', random_state=17) # what's the default criterion in sklearn?
dt.fit(train_scaled, train_target) # what is target variable here?

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target)) # does it show the model is overfitted?


# In[15]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alchol','sugar','ph'])
plt.show()


# In[16]:


get_ipython().run_line_magic('pinfo', 'plot_tree')


# ### Pruning

# In[17]:


# prune to avoid overfitting.
dt = DecisionTreeClassifier(max_depth=4, random_state=17, criterion='log_loss')
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target)) # did this ameliorate overfitting ?


# In[18]:


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


# In[19]:


# now plot the tree!
plt.figure(figsize=(12,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH']) # when filled=True, does the filled color have any meaning ?
plt.show()


# In[20]:


print(dt.feature_importances_) # what does the feature_importance_ show and how, it is used in other modeling?


# In[21]:


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

# In[22]:


dt_prediction = dt.predict(test_input)
print(dt_prediction)


# ### Show Confusion Matrix

# In[23]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(test_target, dt_prediction)
cm_df=pd.DataFrame(cm, columns=['predicted_0', 'predict_1'], index=['actual_0','actual_1'])
print(cm_df)


# In[24]:


plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


# ### Information Gain

# In[25]:


plt.figure(figsize=(5,4))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH']) # what if you do not pass the argument of feature_names?
plt.show()


# In[26]:


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


# In[27]:


initial_entropy - entropy_after_split


# ### Using min_impurity_decrease instead of max_depth

# In[28]:


dt2 = DecisionTreeClassifier(min_impurity_decrease=0.001, random_state=17) #min_impurity_decrease= is used to replace what parameter used above?
dt2.fit(train_input, train_target)

print(dt2.score(train_input, train_target))
print(dt2.score(test_input, test_target))


# ## Entropy vs. Gini Impurity

# $$Entropy(D) = -\sum_{i=1}^{n}p_ilog_2(p_i)$$
# $$Gini(D) = \sum_{i=1}^{n}p_i^2$$

# Gini impurity and entropy are in a monotonic relationship, meaning they increase or decrease together based on the uniformity or imbalance of the probability distribution.
# Gini impurity is more commonly used in decision trees than entropy because it has a lower computational cost.

# In[159]:


import numpy as np
p=np.linspace(0.001,0.999, 1000)
entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
entropy_left= -p * np.log2(p) 
entropy_right= -(1-p) * np.log2(1-p) 

plt.plot(p,-np.log2(p))
# plt.plot(p, entropy_left)
# plt.plot(p, entropy_right)
# plt.plot(p, entropy)

plt.axhline(y=0, color='grey')
plt.xticks([0,1])
plt.ylim(-3,3)
plt.gca().set(xlabel='P: Prob', ylabel='$y=log_2(X)$');


# In[163]:


import numpy as np
import matplotlib.pyplot as plt


plt.plot(p, -np.log2(p))

plt.axhline(y=0, color='grey')
plt.xticks([0, 1])
plt.ylim(-3, 5)
plt.gca().set(xlabel='P: Prob', ylabel='$y=log_2(X)$')
plt.show()


# In[184]:


import numpy as np
import matplotlib.pyplot as plt

p_val = 0.3
p = np.linspace(0.001, 0.999, 1000)
h_blue = -np.log2(p_val)
h_red = -np.log2(1-p_val)
area_blue = p_val * h_blue
area_red = (1-p_val) * h_red

plt.figure(figsize=(10, 6))
plt.plot(p, -np.log2(p), 'purple', linewidth=3)
plt.fill_between([0, p_val], 0, h_blue, alpha=0.3, color='blue')
plt.fill_between([0, 1-p_val], 0, h_red, alpha=0.3, color='red')
plt.plot(p_val, h_blue, 'bo', markersize=12)
plt.plot(1-p_val, h_red, 'ro', markersize=12)
plt.axvline(p_val, color='blue', linestyle='--', linewidth=2, alpha=0.6)
plt.axvline(1-p_val, color='red', linestyle='--', linewidth=2, alpha=0.6)
plt.axhline(0, color='grey')
plt.text(p_val/2, h_blue/2, f'{area_blue:.3f}', 
         fontsize=18, ha='center', fontweight='bold', color='darkblue')
plt.text((1-p_val)/2, h_red/2, f'{area_red:.3f}',
         fontsize=18, ha='center', fontweight='bold', color='darkred')
plt.xlabel('p', fontsize=12)
plt.ylabel(r'$-\log_2(p)$', fontsize=12)
plt.title(f'Binary Entropy: p={p_val}, H={area_blue + area_red:.3f}', 
          fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.2, 2.5)
plt.tight_layout()
plt.show()


# In[191]:


# 더 명확한 예시
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

p_true = 0.3

# 4가지 경우
cases = [
    (0.3, 'Perfect: q=p'),
    (0.5, 'Wrong: q=0.5'),
    (0.7, 'Very wrong: q=0.7'),
    (0.1, 'Very wrong: q=0.1')
]

for idx, (q, title) in enumerate(cases):
    ax = axes[idx//2, idx%2]
    p = np.linspace(0.001, 0.999, 1000)
    
    ax.plot(p, -np.log2(p), 'gray', linewidth=2, alpha=0.3)
    
    # q의 높이 사용
    h_blue = -np.log2(q)
    h_red = -np.log2(1-q)
    
    # p의 너비 사용
    ax.fill_between([0, p_true], 0, h_blue, alpha=0.4, color='blue')
    ax.fill_between([0, 1-p_true], 0, h_red, alpha=0.4, color='red')
    
    ax.plot(q, h_blue, 'bs', markersize=12)
    ax.plot(1-q, h_red, 'rs', markersize=12)
    ax.axvline(p_true, color='black', linestyle='--', linewidth=2)
    
    ce = p_true * h_blue + (1-p_true) * h_red
    h_true = -p_true * np.log2(p_true) - (1-p_true) * np.log2(1-p_true)
    
    color = 'green' if idx == 0 else 'red'
    ax.text(0.5, 2.0, 
            f'True p={p_true}, Use q={q}\n'
            f'CE = {ce:.3f}\n'
            f'Entropy = {h_true:.3f}\n'
            f'Penalty = {ce - h_true:.3f}',
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('p')
    ax.set_ylabel(r'$-\log_2(p)$')
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.2, 2.5)

plt.tight_layout()
plt.show()

print("=" * 60)
print("CROSS ENTROPY 직관")
print("=" * 60)
print("진짜 분포: p = 0.3")
print("-" * 60)
for q, _ in cases:
    ce = -p_true * np.log2(q) - (1-p_true) * np.log2(1-q)
    h = -p_true * np.log2(p_true) - (1-p_true) * np.log2(1-p_true)
    print(f"q={q:.1f}: CE={ce:.3f}, Entropy={h:.3f}, Penalty={ce-h:.3f}")
print("=" * 60)



# In[ ]:




