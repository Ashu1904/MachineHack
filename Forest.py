#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:


train=pd.read_csv('C:/Users/HP/Documents/Ashu/MachineHack/Forest_Cover_participants_Data/train.csv')


# In[4]:


test=pd.read_csv('C:/Users/HP/Documents/Ashu/MachineHack/Forest_Cover_participants_Data/test.csv')


# In[5]:


train.info()


# In[6]:


test.head()


# In[7]:


X=train.drop(columns=['Cover_Type'])
X


# In[8]:


y=train.Cover_Type
y


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


lm=LinearRegression()


# In[13]:


lm.fit(X_train,y_train)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
rfc=RandomForestClassifier(n_estimators=450,criterion='gini',random_state=100)
rfc.fit(X_train,y_train)


# In[15]:


rfc_pred=rfc.predict_proba(X_test)


# In[16]:


rfc.score(X_test,y_test)


# In[17]:


test


# In[18]:


pred_final=rfc.predict_proba(test)


# In[19]:


s=pred_final
s


# In[27]:


output=pd.DataFrame(s)


# In[28]:


output.to_excel('Desktop\Solution.xlsx',index=False)


# In[ ]:




