#!/usr/bin/env python
# coding: utf-8

# In[67]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


BTC = pd.read_csv('btcdata.csv')


# In[69]:


BTC.head()


# In[70]:


BTC.tail()


# In[71]:


BTC['Price'] = BTC['Price'].str.replace(',', '')
BTC['Open'] = BTC['Open'].str.replace(',', '')
BTC['High'] = BTC['High'].str.replace(',', '')
BTC['Low'] = BTC['Low'].str.replace(',', '')
BTC['Vol.'] = BTC['Vol.'].str.replace('.', '')
BTC['Vol.'] = BTC['Vol.'].str.replace('K', '000')
BTC['Vol.'] = BTC['Vol.'].str.replace('M', '000000')
BTC['Change %'] = BTC['Change %'].str.replace('%', '')


# In[72]:


BTC.columns


# In[73]:


BTC.drop('Date',axis=1,inplace=True)


# In[74]:


X = BTC[['Open', 'High', 'Low', 'Vol.', 'Change %']]
y = BTC['Price']


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[77]:


from sklearn.linear_model import LinearRegression


# In[78]:


lm= LinearRegression()


# In[82]:


lm.fit(X_train, y_train)


# In[84]:


#Intercept

print(lm.intercept_)


# In[86]:


#Coeff
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficients'] )
coeff_df


# In[87]:


predictions = lm.predict(X_test)


# In[88]:


plt.scatter(y_test,predictions)


# In[90]:


from sklearn import metrics


# In[92]:


print('MAE:' , metrics.mean_absolute_error(y_test, predictions))
print('MSE:' , metrics.mean_squared_error(y_test, predictions))
print('RMSE:' , np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




