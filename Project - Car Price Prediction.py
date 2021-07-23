#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[67]:


data = pd.read_csv(r'E:/ML Projects/audi.csv')


# In[68]:


data.head()


# In[69]:


data.dtypes


# In[70]:


data.isnull().sum()


# In[71]:


x = data.iloc[:,:-1]


# In[72]:


res = pd.get_dummies(x['model'],drop_first=True)


# In[73]:


res1 = pd.get_dummies(x['transmission'],drop_first=True)
res2 = pd.get_dummies(x['fuelType'],drop_first=True)


# In[74]:


result = pd.concat((res,res1),axis=1)
fin_result = pd.concat((result,res2),axis=1)


# In[75]:


x = pd.concat((fin_result,x),axis=1)


# In[76]:


x = x.drop(['model','transmission','fuelType'],axis=1)


# In[77]:


x['Intercept'] = 1


# In[78]:


VIF = pd.DataFrame()
VIF['Variables'] = x.columns


# In[79]:


VIF['vif'] = [vif(x.values,i) for i in range(x.shape[1])]


# In[80]:


VIF


# In[81]:


x = x.drop(['Intercept'],axis=1)


# In[82]:


x.shape


# In[83]:


x = x.values


# In[84]:


y = data.iloc[:,-1].values


# In[85]:


y.shape


# In[86]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,
                                                random_state=0)


# In[87]:


x_train.shape


# In[88]:


y_train.shape


# In[89]:


classifier = LinearRegression()


# In[90]:


classifier.fit(x_train,y_train)


# In[91]:


y_pred = classifier.predict(x_test)


# In[92]:


r2 = r2_score(y_test,y_pred)


# In[93]:


r2


# In[94]:


adj_r2 = (1-(1-r2)*(8000/79965))


# In[95]:


adj_r2


# In[ ]:




