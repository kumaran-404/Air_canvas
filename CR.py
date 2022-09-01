from tensorflow.keras.datasets import mnist
import numpy as np
from matplotlib import pyplot 
from tensorflow import keras
import tensorflow as tf 
import pickle 


#Load data
(train_digits),(test_digits) = mnist.load_data()


(train_d_x,train_d_y) = train_digits
(test_d_x,test_d_y) = test_digits


train_d_x= train_d_x/255
test_d_x= test_d_x/255

#flatten
train_d_x = train_d_x.reshape(len(train_d_x),784)
test_d_x = test_d_x.reshape(len(test_d_x),784)


#model 



model = keras.Sequential(
    keras.layers.Dense(10,input_shape=(784,),activation="sigmoid")
)
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

 
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )


model.fit(train_d_x,train_d_y,epochs=10)


tf.keras.models.save_model(model,"./model")


