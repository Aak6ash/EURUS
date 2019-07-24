# Modules
import torch
import StackedEsn as etnn
import torch.optim as optim
import torch.nn as nn
import echotorch.utils
from torch.autograd import Variable
import numpy as np
import mdp
import matplotlib.pyplot as py
from sklearn.metrics import mean_squared_error
# Data
import model_data as md

# Data params

batch_size = 100
spectral_radius = 0.9
leaky_rates = [0.7, 0.7, 0.7]
input_dim = 5
output_dim = 1
n_hidden = [100, 50, 25]
epochs = 10
 
# Use CUDA?
use_cuda = False
# use_cuda = torch.cuda.is_available() if use_cuda else False

# Manual seed
'''mdp.numx.random.seed(1)
np.random.seed(2)
torch.manual_seed(1)'''

# Loading Data

# training data
input_data = md.train_in
target_data = md.train_out

# testing data
test_in = md.validation_in[md.start:md.end]
test_out = md.validation_out[md.start:md.end]

# print(input_data.shape, target_data.shape)
# print(test_in.shape, test_out.shape)

# ESN cell 
esn = etnn.StackedESN(input_dim=input_dim, hidden_dim=n_hidden, output_dim=output_dim, leaky_rate=leaky_rates, spectral_radius=spectral_radius)

if use_cuda:    
    esn.cuda()

# end if

criterion = nn.MSELoss()

optimizer = optim.Adagrad(esn.parameters())

# Stochastic Gradient Descent

l = []

for epoch in range(epochs):
    # Iterate over batches
    print("Epoch ", epoch, ":")

    l_avg = 0

    for i in range(0, int(md.n_training/batch_size)):

        # To variable
        # print("Batch:", i)

        input, target = torch.FloatTensor(input_data[i*batch_size:(i+1)*batch_size]), torch.FloatTensor(target_data[i*batch_size:(i+1)*batch_size])

        # print(input.shape,target.shape)

        input, target = Variable(input), Variable(target)
        
        if use_cuda:
            input, target = input.cuda(), target.cuda()

        optimizer.zero_grad()

        # Forward

        out = esn(input)

        # print(out.shape)    
        # print(target.shape)

        loss = criterion(out, target)

        l_avg = loss + l_avg

        # Backward pass 
    
        loss.backward() 

        # Optimize
        optimizer.step()

        # print(u"Train MSE: {}".format(float(loss.data)))

    l_avg = l_avg / (batch_size)

    l.append(l_avg)

    print(l_avg)
        
    # end for

test_in, test_out = torch.FloatTensor(test_in), torch.FloatTensor(test_out)
if use_cuda: 
    test_in, test_out = test_in.cuda(), test_out.cuda()

z = esn(test_in)
z = z.cpu().detach().numpy()
print(z.shape)
z = np.squeeze(z)
test_out = np.squeeze(np.array(test_out))
delta = np.squeeze(np.array((test_out-z)*md.max[0]))
mse = mean_squared_error(test_out*md.max[0], z*md.max[0])
print(np.amax(delta), np.amin(delta), np.mean(np.absolute(delta)))
print(mse)
py.plot(z*md.max[0])
py.plot(test_out*md.max[0])
py.plot(delta)

# py.plot(l)

py.show()