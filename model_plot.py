import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import model as m
import model_data as md

x = [i for i in range(md.out_step) ]
# x = [i for i in range(md.end-md.start)]

'''md.test_out = np.squeeze(md.test_out)*md.max[0]

pto = []
for i in range(len(md.test_out)):
        pto = np.append(pto, md.test_out[i][0])'''
# print(md.validation_out[md.start:md.end].shape)

y = np.squeeze(m.z)
# t = np.squeeze(m.pz)
# t1 = np.squeeze(m.pd)

y1 = np.squeeze(md.validation_out[md.start:md.end]*md.max[0])
y2 = np.squeeze(m.delta)

# mse = mean_squared_error(pto, m.pz)
# print(mse)

plt.plot(x,y,label='Prediction')
plt.plot(x,y1,label='Real')
plt.plot(x,y2,label='Delta')

plt.legend()

plt.show()

