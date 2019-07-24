import pandas as pd
import numpy as np
from matplotlib import pyplot as py
from statsmodels.tsa.seasonal import seasonal_decompose


d = pd.read_csv("weatherHistory.csv",usecols=["Temperature","Humidity","Pressure","WindSpeed","WindBearing ","Visibility ","ApparentTemperature"])
data = d.as_matrix()

n_features = 5
in_step = 24
out_step = 24
fraction = 0.7
step = 1
sub_seq = 1
sub_step = in_step//sub_seq


n_training=int(96432*fraction)
n_validation=96432-n_training

for i in range(96453):
        data[i][1] = data[i][0]-data[i][1]
        if data[i][3] == 0:
                data[i][3] = data[i-1][3] 

data=data[:-21] 

max=np.amax(data,axis=0)
min=np.amin(data,axis=0)
max=max-min
data=np.divide(data,max)


def data_splitter(start,end,data,in_step,out_step,step):
        x,y=list(),list()
        for i in range(start,end,step):
                x_seq=data[i:i+in_step,[0,2,3,4,6]]
                y_seq=data[i+in_step:i+in_step+out_step,0]
                x.append(x_seq)
                y.append(y_seq)
        return np.array(x),np.array(y)


#getting seq
train_in,train_out=data_splitter(0,n_training,data,in_step,out_step,step)

#reshaping
train_in=train_in.reshape(train_in.shape[0],sub_step,n_features)
train_out=train_out.reshape(train_out.shape[0],train_out.shape[1],1)

print(train_in.shape)

#getting the seq
validation_in,validation_out=data_splitter(n_training,n_validation+n_training-168*3,data,in_step,out_step,step)

#reshaping
validation_in=validation_in.reshape(validation_in.shape[0],sub_step,n_features)
validation_out=validation_out.reshape(validation_out.shape[0],out_step,1)

# print(validation_out.shape)
start = 0
end = 1000
test_in,test_out=validation_in[start:end],validation_out[start:end]

test_in=test_in.reshape(test_in.shape[0],sub_step,n_features)

# test_out=test_out.reshape(test_out.shape[0],test_out.shape[1])
# print(test_in.shape,test_out.shape)

'''py.plot(np.squeeze(test_out*max[0]))
py.show()'''
