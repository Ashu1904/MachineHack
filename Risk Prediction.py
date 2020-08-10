#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:


train=pd.read_csv('C:/Users/HP/Documents/Ashu/MachineHack/Financial_Risk_Participants_Data/train.csv')


# In[4]:


train.head()


# In[5]:


test=pd.read_csv('C:/Users/HP/Documents/Ashu/MachineHack/Financial_Risk_Participants_Data/test.csv')


# In[6]:


test.head()


# In[7]:


train.info()


# In[8]:


test.info()


# In[9]:


X=train.drop(columns=['IsUnderRisk'])
X


# In[10]:


y=train.IsUnderRisk
y


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[15]:


predictions = logmodel.predict_proba(X_test)


# In[16]:


logmodel.score(X_test,y_test)


# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix


# In[19]:


rfc=RandomForestClassifier(n_estimators=100,criterion='gini',random_state=100)


# In[20]:


rfc.fit(X_train,y_train)


# In[21]:


rfc_pred=rfc.predict_proba(X_test)


# In[22]:


rfc.score(X_test,y_test)


# In[23]:


pred_final=rfc.predict_proba(test)


# In[24]:


pred_final


# In[25]:


pred_final.shape


# In[26]:


output=pd.DataFrame(pred_final)


# In[28]:


output.to_excel(r'C:\Users\HP\Documents\Ashu\MachineHack\Financial_Risk_Participants_Data\Solution.xlsx',index=False)


# In[ ]:




