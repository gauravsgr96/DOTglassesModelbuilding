#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


df= pd.read_csv(r"C:\Users\DELL\Desktop\DOT glasses.csv")


# In[24]:


print(df)


# In[25]:


df.head()


# In[26]:


df.tail()


# In[27]:


df.isnull()


# In[28]:


df.isnull().sum()


# In[29]:


#Basic information

df.info()


# In[30]:



#Describe the data

df.describe()


# In[33]:



# Load the data into a DataFrame
df = pd.read_csv(r'C:\Users\DELL\Desktop\DOT glasses.csv')

# Create scatter plots
plt.scatter(df['TV'], df['Sales'])
plt.xlabel('TV advertising ($1000s)')
plt.ylabel('Sales ($1000s)')
plt.show()

plt.scatter(df['Radio'], df['Sales'])
plt.xlabel('Radio advertising ($1000s)')
plt.ylabel('Sales ($1000s)')
plt.show()

plt.scatter(df['Newspaper'], df['Sales'])
plt.xlabel('Newspaper advertising ($1000s)')
plt.ylabel('Sales ($1000s)')
plt.show()


# In[34]:



# Create a correlation matrix
corr = df.corr()

# Create a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# In[35]:


plt.hist(df['Sales'])
plt.xlabel('Sales ($1000s)')
plt.ylabel('Frequency')
plt.show()


# In[36]:



import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[37]:


data = pd.read_csv(r"C:\Users\DELL\Desktop\DOT glasses.csv")


# In[38]:


X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[39]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[40]:


X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())


# In[ ]:




