# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:12:47 2018

@author: PAVEETHRAN
"""
#OM

import numpy
import tensorflow as tf
import keras
import tensorflow.python
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
#INITIALSE CNN
cl=Sequential()

#ADD INPUT LAYER
cl.add(Conv2D(32,(3,3),activation='relu',input_shape=[64,64,3])) 
#ADD THE MAX POOLING LAYER
cl.add(MaxPooling2D(pool_size=(2,2)))#...same padding is zero padding
#layer 2
cl.add(Conv2D(32,(3,3),activation='relu'))
cl.add(MaxPooling2D(pool_size=(2,2)))
#layer 3
cl.add(Conv2D(64,(3,3),activation='relu'))
cl.add(MaxPooling2D(pool_size=(2,2)))
#FLATTEN
cl.add(Flatten()) 

#ADING THE FIRST LAYER OF DENSE..to maka a fully connected cnn
cl.add(Dense(activation='relu',output_dim=64))
cl.add(Dropout(0.5))
#O/P LAYER
#cl.add(Dense(activation='softmax',output_dim=3))
cl.add(Dense(3))
cl.add(Activation(tf.nn.softmax))
"""
cl.add(Conv2D(32, (3, 3), input_shape=[64,64,3]))
cl.add(Activation('relu'))
cl.add(MaxPooling2D(pool_size=(2, 2)))

cl.add(Conv2D(32, (3, 3)))
cl.add(Activation('relu'))
cl.add(MaxPooling2D(pool_size=(2, 2)))

cl.add(Flatten())
cl.add(Dense(64))
cl.add(Activation('relu'))
cl.add(Dropout(0.5))
cl.add(Dense(3))
cl.add(Activation('sigmoid'))
"""
#Compiling the cnn
cl.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#IMPLEMENT THE TRAIN AND TEST IMAGES IN CNN
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#LETS DO THISSSSS
trainingset = train_datagen.flow_from_directory('training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,save_to_dir='training_set',
                                                    class_mode='categorical')

                                                    
testset = test_datagen.flow_from_directory( 'test_set',
                                            target_size=(64, 64),
                                            batch_size=32,save_to_dir='training_set',
                                            class_mode='categorical')

#HERE WE GO
cl.fit_generator(trainingset,steps_per_epoch=10,nb_epoch=5,validation_data=testset,validation_steps=60)

#MAKE PREDICTION FOR A SINGLE IMAGE
import numpy as np
trainingset.class_indices
img=load_img('r.jpg',target_size=(64,64))
#img = [np.exand_dims(img, 1) if img is not None and img.ndim == 1 else img]
img=img_to_array(img)
img=np.expand_dims(img,axis=0)
pred=cl.predict(img)

threshold=tf.constant([0.25,0.25,0.25],dtype=tf.float32)
raw_prob=tf.greater_equal(pred,threshold)
    
#{'Chihuahua': 0, 'Yorkshire_terrier': 1, 'pug': 2}
if raw_prob[0][0] or raw_prob[0][1] or raw_prob[0][2]==True:
    if pred[0][0]==1:
        pred='Chihuahua';
    elif pred[0][1]==1:
        pred='Yorkshire Terrier';   
    elif pred[0][2]==1:
        pred='Pug';
    

    
#ORR..ORR....ORR...ORR
#img=np.reshape(img,(1,64,64,3))
#cl.predict(img)
#SEE WHAT IS BOTTLENECK FEATURE