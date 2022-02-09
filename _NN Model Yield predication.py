#!/usr/bin/env python
# coding: utf-8

# # Neural Network for Yield Predication 
# 

# In[1]:


import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np

import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt


# # Loading the data

# In[2]:


# we are using pandas libaray for reading the file
# We are reading ecxel file and path should be where you ecxel file is located.
# place "r" before the path string to address special character,such as '\'.
# Don't forget to put the file name at the end of the path + '.xlsx'

data=pd.read_excel(r"C:\python\Yield Predicition\plain 1.xlsx")


# In[3]:


#Showing the first five entries of data
data.head()


# In[4]:


# As there some extra columns so we need to perform data cleaning

data.drop(columns='Month', inplace=True)
data.drop(columns='NRDE', inplace=True)
data.drop(columns='Soil Fertility', inplace=True)
data.head()


# In[5]:


# Information about your data, how many entries and which type of data is present in dataframe
data.info()


# In[6]:


#tells the shape of dataframe
data.shape


# In[7]:


data.plot(kind='scatter', x='Rainfall', y='Yield', figsize=(6,6))


# In[8]:


x=data.iloc[:, 0:13]
x.head()


# In[9]:


y=data.iloc[:, -1]
y.head()


# In[10]:


# Split the data into input (x) training and testing data, and ouput (y) training and testing data, 
# with training data being 80% of the data, and testing data being the remaining 20% of the data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[11]:


X_train.shape


# In[12]:


y_train.shape


# In[13]:


X_test.shape


#  # Scaling

# In[14]:


#train_labels = np.array(y_train)
#train_samples = np.array(X_train)
#train_labels, train_samples = shuffle(train_labels, train_samples)


# In[15]:


#scaler = MinMaxScaler(feature_range=(0,1))
#scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))


# In[16]:


# Scale both training and testing input data

X_train = preprocessing.scale(X_train)

X_test = preprocessing.scale(X_test)


# # Defines model

# In[ ]:





# In[17]:


...
# define the keras model
model = Sequential()
model.add(Dense(13, input_dim=13, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(42, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(22, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(22, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='linear'))


# In[18]:


model.summary()


# # Compiling the model

# In[19]:


model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


# # Training the model

# In[20]:


history=model.fit(x=X_train, y=y_train, batch_size=2, epochs=3000, verbose=2)


# # Ploting the loss

# In[21]:


# plot loss during training
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# # Evaluate the model on test set
# 

# In[23]:


y_train_pred= model.predict(X_train)
y_predicted = model.predict(X_test)


error= mean_absolute_error(y_test,y_predicted)
print('MAE: %.3f' % error)

error1= mean_squared_error(y_test, y_predicted)
print('MAE: %.3f' % error1)


# In[24]:


plt.plot(y_train, y_train_pred,'*r')
plt.plot(y_test, y_predicted, '*g')
plt.figure()






