import os
import sys
import numpy as np
import matplotlib.pyplot as plt
basePath = "./results"

def joinPath(arr):
	return '/'.join(arr) + '/'

def returnList(dir):
	return np.array(os.listdir(dir))

def plot(data):
	bins = range(len(data[0]))
	ranges = (data.min(),data.max())
	plt.hist(data[0],bins,ranges)
	plt.hist(data[1],bins,ranges)
	plt.legend(['dense','focused'])
	plt.show()

for folder in returnList(basePath)[[1]]:
	dir = joinPath([basePath, folder])
	data=[]
	print(dir)
	for file in returnList(dir)[[2,5]]:
		f = dir+file
		data.append(np.load(f, allow_pickle=True))
	print(data)
	break
	plot(np.array(data)[:,-100:])
