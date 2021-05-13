#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff


# ### Importing DataSet and gaining some Information

# In[2]:


# Importing dataset and examining it
dataset = pd.read_csv("Clients.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


# ### Converting Categorical Data into Numeric

# In[3]:


# Converting Categorical features into Numerical features
dataset['job'] = dataset['job'].map({'admin.':0, 'blue-collar':1, 'entrepreneur':2, 'housemaid':3, 'management':4, 'retired':5, 'self-employed':6, 'services':7, 'student':8, 'technician':9, 'unemployed':10})
dataset['marital'] = dataset['marital'].map({'divorced':0, 'married':1, 'single':2})
dataset['education'] = dataset['education'].map({'primary':0, 'secondary':1, 'tertiary':2})
dataset['default'] = dataset['default'].map({'no':0, 'yes':1 })
dataset['housing'] = dataset['housing'].map({'no':0, 'yes':1})
dataset['personal'] = dataset['personal'].map({'no':0, 'yes':1})
dataset['term'] = dataset['term'].map({'no':0, 'yes':1})


# ### Correlation and Causation

# In[4]:


print(dataset.info())
# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')


# ### Divding the Variables into Subsets

# In[6]:


X = dataset
#Personal Data
subset1 = X[['age','job','marital','education']]

#Loan Data
subset2 = X[['age','balance','housing','personal','term','default']]


# ### Normalizing the numeric variables of subsets

# In[7]:


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)


# In[9]:


# Analysis on subset1 - Personal Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[10]:


# Analysis on subset2 - Loan Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[11]:


# Running KMeans to generate labels for X1
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)


# In[12]:


# Running KMeans to generate labels for X2
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X2)


# ### T-SNE for X1

# In[13]:


# Implementing t-SNE to visualize dataset for X1
tsne = TSNE(n_components = 2, perplexity =30,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

age = list(X['age'])
job = list(X['job'])
marital = list(X['marital'])
education = list(X['education'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'age: {a}; job: {b}; marital:{c}; education:{d}' for a,b,c,d in list(zip(age,job,marital,education))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')


# ### T-SNE for X2

# In[ ]:


# Implementing t-SNE to visualize dataset for X2
tsne = TSNE(n_components = 2, perplexity =30,n_iter=2000)
x_tsne = tsne.fit_transform(X2)

age = list(X['age'])
balance = list(X['balance'])
housing = list(X['housing'])
personal = list(X['personal'])
term = list(X['term'])
default = list(X['default'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'age: {a}; balance: {b}; housing:{c}; personal:{d}; term:{e};default:{f}' for a,b,c,d,e,f in list(zip(age,balance,housing,personal,term,default))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')


# In[ ]:




