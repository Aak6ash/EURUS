import numpy as np
import torch

res_size = 5
size = (res_size, res_size)
sparsity = 0.1
input_set = [-1.0, 1.0]
p = np.append([1.0 - sparsity], [sparsity / len(input_set)] * len(input_set))
w = np.random.choice(np.append([0], input_set), size, p=np.append([1.0 - sparsity], [sparsity / len(input_set)] * len(input_set)))
w = (np.random.randint(0, 2, size) * 2.0 - 1.0)
leak = torch.DoubleTensor(1).fill_(0.8)
print(leak)
print(w)