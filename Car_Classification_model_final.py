#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import time
import numpy as np
import pickle
from keras.preprocessing import image
from matplotlib import pyplot
from PIL import Image
import glob
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


# In[2]:


image_list_SUV = []  ## reading images of sedan from path and storing in this list
image_list_sedan = []  ## reading images of sedan from path and storing in this list

## reading the images from the source and saving it in the list after resizing it to 224,224
 ## image size for input to mobilenet is 224 x 224
for filename in glob.glob('car_data/car_data/SUV/*.jpg'): #assuming gif
    im= cv2.imread(filename)
    image_list_SUV.append(cv2.resize(im,(224,224))) 
for filename in glob.glob('car_data/car_data/sedan/*.jpg'): #assuming gif
    im= cv2.imread(filename)
    image_list_sedan.append(cv2.resize(im,(224,224)))


# In[3]:


## Dataset preparation

## The images for sedan and suv are combined and sedan is given label as 0 and SUV is given label as 1.

x = np.array(image_list_sedan[:500])    
x = np.concatenate((x,np.array(image_list_SUV[:500])))
y = np.zeros(len(image_list_sedan[:500]))
y = np.concatenate((y, np.ones(len(image_list_SUV[:500]))))


# In[4]:


## Split the data into train and test dataset

x_scaled = x / 255  ## normalizing the images

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.30, shuffle=True)


# In[5]:


## This model gives all the layer except the last layer
feature_ext_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_freeze = hub.KerasLayer(
    feature_ext_model, input_shape=(224, 224, 3), trainable=False)

## Combining the base model of MobileNet without the top layer and adding our own layer to classifiy two classes
classes_num = 2  ## classifiy two classes- sedan or SUV
model_classifier = tf.keras.Sequential([pretrained_model_freeze, tf.keras.layers.Dense(classes_num)])

model_classifier.summary()
model_classifier.compile(optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])


# In[6]:


model_classifier.fit(x_train, y_train, epochs=10)
model_classifier.evaluate(x_test, y_test)


# In[8]:


model_classifier.evaluate(x_test, y_test)


# In[9]:


model_classifier.save('class_model_updated') ## Saving the classifier model to reuse it


# In[ ]:





# In[ ]:





# In[ ]:




