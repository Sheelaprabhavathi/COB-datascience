#!/usr/bin/env python
# coding: utf-8

# # import packages

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv("netflix1.csv")
df


# In[11]:


df['date_added'] = pd.to_datetime(df['date_added'], format='%m/%d/%Y')


# In[12]:


df['date_added'] = pd.to_datetime(df['date_added'], format='%d-%m-%Y')


# In[13]:


df['date_added']=df['date_added'].dt.strftime('%d-%m-%Y')
df


# In[15]:


df.type.value_counts()


# In[16]:


sns.countplot(x='type',data=df)
plt.title("count VS Type of Shows")


# In[17]:


df['country'].value_counts().head(10)


# In[19]:


plt.figure(figsize=(12,6))
sns.countplot(y='country',order=df['country'].value_counts().index[0:10],data=df)
plt.title('country wise content on NetFlix')


# In[20]:


movie_countries=df[df['type']=='Movie']
tv_show_countries=df[df['type']=='Tv Show']
plt.figure(figsize=(12,6))
sns.countplot(y='country',order=df['country'].value_counts().index[0:20],data=movie_countries)
plt.title('Top 10 countries producing movies on Netflix')


# In[21]:


df.rating.value_counts()


# In[22]:


plt.figure(figsize=(10,6))
sns.countplot(x='rating',order=df['rating'].value_counts().index[0:10],data=df)
plt.title('Rating of Shows on Netfix VS Count')


# In[23]:


df.release_year.value_counts()[:20]


# In[24]:


plt.figure(figsize=(9,6))
sns.countplot(x='release_year',order=df['release_year'].value_counts().index[0:20],data=df)
plt.title('Content Release in years on Netfix VS Count')


# In[26]:


plt.figure(figsize=(12,8))
sns.countplot(y='listed_in',order=df['listed_in'].value_counts().index[0:20],data=df)
plt.title('Top 20 Genre on Netfix')


# In[ ]:




