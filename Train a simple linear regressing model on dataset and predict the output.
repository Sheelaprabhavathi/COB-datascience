#!/usr/bin/env python
# coding: utf-8

# # Train Dataset

# In[73]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[74]:


Train_url='https://docs.google.com/spreadsheets/d/e/2PACX-1vRTK2NvcndgPX41Czu6Ft2Ho_nE-z50BgTqdzwFW0rsJ2nvyNLe2DoIg1COzUbgw80oaRBjfy5-WtFk/pubhtml?urp=gmail_link'


# In[75]:


response=requests.get(Train_url)
response.raise_for_status()
soup=BeautifulSoup(response.text,"html.parser")
s=soup.find('table',{'class':'waffle'})
tr=s.find_all('tr')
x=[0,0]
y=[]
for i in tr:
    x=i.find_all('td',{'class':'s1'})
    if len(x)>1:
        y.append({
            'x':int(x[0].text.strip()),
            'y':x[1].text.strip()
        })


# In[76]:


df=pd.DataFrame(y)
df['y']=pd.to_numeric(df['y'])
df.to_csv('Train.csv',index=False)
print("saved to train.csv")


# In[77]:


plt.xlabel('x')
plt.ylabel('y')
plt.scatter(df['x'],df['y'])
plt.show()


# In[78]:


data=pd.read_csv('train.csv')
data.head()


# In[79]:


x_train=data[['x']]
x_train
y_train=data['y']
y_train


# # Test Dataset

# In[80]:


from sklearn import linear_model
model=linear_model.LinearRegression()
model.fit(x_train,y_train)


# In[81]:


Test_url='https://docs.google.com/spreadsheets/d/e/2PACX-1vRyvZ7lknwiSghK9aen1SaTEYoN3JS40rrGLpcyrsVZy1tB2T4gn6Y3-cdzPUFCPMmmqREWefW3kl4_/pubhtml'


# In[82]:


response=requests.get(Test_url)
response.raise_for_status()
soup=BeautifulSoup(response.text,"html.parser")
s=soup.find('table',{'class':'waffle'})
tr=s.find_all('tr')
x=[0,0]
y=[]
for i in tr:
    x=i.find_all('td',{'class':'s1'})
    if len(x)>1:
        y.append({
            'x':int(x[0].text.strip()),
            'y':x[1].text.strip()
        })


# In[83]:


df=pd.DataFrame(y)
df['y']=pd.to_numeric(df['y'])
df.to_csv('Test.csv',index=False)
print("saved to test.csv")


# In[84]:


x_test=data[['x']]
x_test
y_test=data['y']
y_test


# In[85]:


y_pred=model.predict(x_test)
y_pred


# In[86]:


plt.scatter(x_test,y_test,color='blue',label='Output')
plt.plot(x_test,y_pred,color='red',linewidth=2,label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()


# In[87]:


r2=r2_score(y_test,y_pred)
print("R.squarel (r2) Score:",r2)

