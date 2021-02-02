#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")df.columns


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df['Unnamed: 32']


# In[8]:


df = df.drop("Unnamed: 32", axis=1)


# In[9]:


df.head()


# In[10]:


df.columns


# In[11]:


df.drop('id', axis=1, inplace=True)
# df = df.drop('id', axis=1)


# In[12]:


df.columns


# In[13]:


type(df.columns)


# In[14]:


l = list(df.columns)
print(l)


# In[15]:


features_mean = l[1:11]

features_se = l[11:21]

features_worst = l[21:]


# In[16]:


print(features_mean)


# In[17]:


print(features_se)


# In[18]:


print(features_worst)


# In[19]:


df.head(2)


# In[20]:


df['diagnosis'].unique()
# M= Malignant, B= Benign


# In[21]:


sns.countplot(df['diagnosis'], label="Count",);


# In[22]:


df['diagnosis'].value_counts()


# In[23]:


df.shape


# In[24]:


df.describe()
# summary of all the numeric columns


# In[25]:


len(df.columns)


# In[26]:


# Correlation Plot
corr = df.corr()
corr


# In[27]:


corr.shape


# In[28]:


plt.figure(figsize=(8,8))
sns.heatmap(corr);


# In[29]:


df.head()


# In[30]:


df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})


# In[31]:


df['diagnosis'].unique()


# In[32]:


X = df.drop('diagnosis', axis=1)
X.head()


# In[33]:


y = df['diagnosis']
y.head()


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[35]:


df.shape


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# In[38]:


y_train.shape


# In[39]:


y_test.shape


# In[40]:


X_train.head(1)


# In[41]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[42]:


X_train


# In[43]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[44]:


y_pred = lr.predict(X_test)


# In[45]:


y_pred


# In[46]:


y_test


# In[47]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[48]:


lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)


# In[49]:


results = pd.DataFrame()
results


# In[50]:


tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[51]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[52]:


y_pred = dtc.predict(X_test)
y_pred


# In[53]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[54]:


dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)


# In[55]:


tempResults = pd.DataFrame({'Algorithm':['Decision tree Classifier Method'], 'Accuracy':[dtc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[56]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[57]:


y_pred = rfc.predict(X_test)
y_pred


# In[58]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[59]:


rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)


# In[60]:


tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[61]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)


# In[62]:


y_pred = svc.predict(X_test)
y_pred


# In[63]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[64]:


rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)


# In[65]:


tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[66]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)


# In[67]:


y_pred = svc.predict(X_test)
y_pred


# In[68]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[69]:


svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)


# In[70]:


tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# In[ ]:




