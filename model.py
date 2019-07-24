#Calc
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
#CNN
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
#LSTM
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import regularizers
#Model
from keras import backend as K
from keras import models as mod
import tensorflow as tf
K.tensorflow_backend._get_available_gpus()
#Data
import model_data as md

'''model=Sequential()
# CNN
model.add(TimeDistributed(Conv1D(filters=5, kernel_size=6, strides=1, activation='tanh', input_shape=(md.in_step,5))))

# model.add(TimeDistributed(MaxPooling1D(pool_size=3))) 

model.add(TimeDistributed(Flatten()))

# LSTM
model.add(LSTM(100,activation='tanh',stateful=False))

model.add(RepeatVector(md.out_step))

model.add(Dense(200,activation='tanh'))
 
model.add(LSTM(200,activation='tanh',return_sequences=True))

model.add(Dense(200,activation='tanh'))

model.add(LSTM(200,activation='tanh',return_sequences=True))

model.add(Dense(100,activation='tanh'))

model.add(TimeDistributed(Dense(1,activation='tanh')))

model.compile(optimizer='Adadelta', loss='msle',metrics=['mse'])    
    
model.fit(md.train_in,md.train_out,epochs=100,verbose=1,batch_size=100,validation_data=(md.validation_in,md.validation_out),shuffle=True)  

model.summary()

model.save("n_model6")'''

model = mod.load_model("n_model2")

#result
z = model.predict(md.test_in)

#print(np.multiply(z,39.905))
#print(np.amax(z)*39.905,np.min(z)*39.905)
#print(np.amax(A4_datahandling.test_out)*39.905,np.min(A4_datahandling.test_out)*39.905)
#print(np.multiply(A4_datahandling.test_out,39.905))
delta = np.multiply(md.test_out,md.max[0])-np.multiply(z,md.max[0])
# print(delta)
print(np.amax(delta),np.amin(delta),np.mean(np.absolute(delta)))
z = np.squeeze(z*md.max[0])
test_out = np.squeeze(md.test_out*md.max[0])
delta = np.squeeze(delta)
mse = mean_squared_error(test_out,z)
print(mse)
# print(z.shape)
# print(delta.shape)
'''pz = []
pd = []
for i in range(len(z)):
    pz = np.append(pz, z[i][0])
    pd = np.append(pd, delta[i][0])'''
