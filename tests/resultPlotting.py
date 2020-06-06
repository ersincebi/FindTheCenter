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

for folder in returnList(basePath):
	dir = joinPath([basePath, folder])
	data=[]
	for file in returnList(dir):
		if 'scores' in file:
			f = dir+file
			loadedData = np.load(f, allow_pickle=True)
			print(loadedData)
			x=input()
			# data.append(loadedData)
	# plot(np.array(data)[:,-100:])
