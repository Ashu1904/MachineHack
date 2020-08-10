#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[6]:


train=pd.read_csv('C:/Users/HP/Documents/Ashu/MachineHack/CCPP_participants_Data/Train.csv')


# In[7]:


test=pd.read_csv('C:/Users/HP/Documents/Ashu/MachineHack/CCPP_participants_Data/Test.csv')


# In[8]:


train


# In[9]:


train.info()


# In[82]:


train['AP']=train['AP'].divide(100)


# In[83]:


X=train.drop(columns=['PE'])
X


# In[84]:


y=train.PE
y


# In[85]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[86]:


from sklearn.ensemble import RandomForestRegressor


# In[87]:


reg1=RandomForestRegressor(n_estimators=2000,criterion='mse',random_state=0,oob_score=False)


# In[88]:


reg1.fit(X_train,y_train)


# In[89]:


reg1.score(X_test,y_test)


# In[90]:


test


# In[91]:


pred_final=reg1.predict(test)


# In[92]:


pred_final


# In[93]:


output=pd.DataFrame({'PE':pred_final})


# In[94]:


output.to_csv(r'C:\Users\HP\Documents\Ashu\MachineHack\CCPP_participants_Data\Solution.csv',index=False)


# In[ ]:




