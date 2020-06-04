import numpy as np
with open('./results/CartPole-v1/CartPole-v1-dense-replay-1590425143.npy', 'rb') as f:
	data = np.load(f, allow_pickle=True)

print(data[0])