import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
from scipy.integrate import trapz
from scipy.stats import ttest_ind

basePath = "./results"
data={}
loss = {}
def joinPath(arr):
	return '/'.join(arr)

def returnList(dir):
	return np.array(os.listdir(dir))

def lossComparison(name, results):
	plt.plot(results, label=name+" loss")
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(loc="upper left")
	
def showTrend(scores, name=''):
	x = []
	y = []
	for i in range(0, len(scores)):
		x.append(int(i))
		y.append(int(scores[i]))

	plt.plot(x, y, label=name+" score per run")

	trend_x = x[1:]
	z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
	p = np.poly1d(z)
	plt.plot(trend_x, p(trend_x), linestyle="-.", label=name+" trend")


	plt.xlabel("episodes")
	plt.ylabel("scores")
	plt.legend(loc="upper left")

def plot(folder):
	dir = joinPath([basePath, folder[0]])
	f = ''
	for file in returnList(dir):

		selectFolder = joinPath([dir,file])
		selectFile = returnList(selectFolder)[[3]]
		f = joinPath([selectFolder,selectFile[0]])

		selectLossFile = returnList(selectFolder)[[1]]
		l = joinPath([selectFolder,selectLossFile[0]])

		data[file] = np.load(f)
		loss[file] = np.load(l)

		d = np.sort(data[file])
	# 	print(f'Area Under {file} Curve',trapz(d))

	# 	fit = stats.norm.pdf(d, np.mean(d), np.std(d))
	# 	pl.plot(d, fit,'-o')
	# 	pl.hist(d, density=True)

	# pl.legend(data.keys())
	# plt.savefig('./plots/histograms/'+folder[0]+'_histogram.png', bbox_inches="tight")


# histogram display over score table on a problem
selectedFolder = returnList(basePath)[[0]]
# pl.figure(figsize=(20,15))
plot(selectedFolder)
# pl.show()

#This part will run after selecting right focused values againts dense values
# trend comparison
selectedDenseScore = data['dense']
selectedFocusedScore = data['focused-02']
selectedDenseLoss = loss['dense']
selectedFocusedLoss = loss['focused-02']
plt.figure(figsize=(20, 15))
showTrend(name=selectedFolder[0]+' dense',scores=selectedDenseScore)
showTrend(name=selectedFolder[0]+' focused',scores=selectedFocusedScore)
plt.savefig('./plots/histograms/'+selectedFolder[0]+'_trend.png', bbox_inches="tight")
plt.show()

# # loss comparison
plt.figure(figsize=(20, 15))
lossComparison(name=selectedFolder[0]+' dense',results=selectedDenseLoss)
lossComparison(name=selectedFolder[0]+' focused',results=selectedFocusedLoss)
print(f'Area Under dense Curve',trapz(selectedDenseLoss))
print(f'Area Under focused Curve',trapz(selectedFocusedLoss))
plt.savefig('./plots/histograms/'+selectedFolder[0]+'_loss.png', bbox_inches="tight")
plt.show()

# t-test will applied according to our hypotesis
# 	we will make an assumption according to area under curve
# 	and will say like this focused nn learn to solve
# 	the problem more fast, accurate and efficient
statistic, pvalue = ttest_ind(selectedDenseScore,selectedFocusedScore)
print('statistic: ',float(statistic))
print('pvalue: ',float(pvalue))