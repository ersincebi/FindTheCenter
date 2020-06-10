import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
from scipy.integrate import trapz
from scipy.stats import ttest_ind
from scipy import ndimage
import matplotlib as mpl

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

def paper_fig_settings(addtosize=0):
	#mpl.style.use('seaborn-white')
	mpl.rc('figure',dpi=144)
	mpl.rc('text', usetex=False)
	mpl.rc('axes',titlesize=16+addtosize)
	mpl.rc('xtick', labelsize=12+addtosize)
	mpl.rc('ytick', labelsize=12+addtosize)
	mpl.rc('axes', labelsize=14+addtosize)
	mpl.rc('legend', fontsize=10+addtosize)
	mpl.rc('image',interpolation=None)
	#plt.rc('text.latex', preamble=r'\usep

def plot(d, mean, std, file):
	fit = stats.norm.pdf(d, mean, std)
	hist, bin_edges = np.histogram(d, bins=np.linspace(d.min(),d.max(),100))
	plt.plot(d, fit,'-o', color=color[file])
	plt.hist(hist, bins=bin_edges, density=True, color=color[file], alpha=0.5)


def bringData(folder, dataFolder, scorePosition, lossPosition, pickle=False ,save=False):
	dir = joinPath([basePath, dataFolder, folder[0]])

	f = ''
	areaCompare = float('-inf')
	fileName = ''

	paper_fig_settings()
	plt.figure(figsize=(20,15))

	for file in returnList(dir):

		selectFolder = joinPath([dir,file])
		selectFile = returnList(selectFolder)[[scorePosition]]
		f = joinPath([selectFolder,selectFile[0]])

		selectLossFile = returnList(selectFolder)[[lossPosition]]
		l = joinPath([selectFolder,selectLossFile[0]])

		data[file] = np.load(f, allow_pickle=pickle)
		loss[file] = np.load(l, allow_pickle=pickle)

		d = np.sort(data[file])

		area = trapz(d)
		mean = np.mean(d)
		std = np.std(d)

		# if areaCompare < area:
		# 	areaCompare = area
		fileName = 'focused'
		try:
			selectedDenseScore = data['dense']
			selectedFocusedScore = data[fileName]
			selectedDenseLoss = loss['dense']
			selectedFocusedLoss = loss[fileName]
		except:
			pass

		selectedFolder = folder[0]

		print(f'{folder[0]} -> {file} -> Curve Area: {area}, Mean: {mean}, std: {std}',)
		
		plot(d, mean, std, file)
	plt.legend(data.keys())
	if save:
		plt.savefig('./plots/histograms/'+selectedFolder+'_histogram.png', bbox_inches="tight")

	print('selected sigma: ', fileName)

	#This part will run after selecting right focused values againts dense values
	# trend comparison
	plt.figure(figsize=(20, 15))
	paper_fig_settings()
	showTrend(scores=selectedDenseScore,name=selectedFolder+' dense')
	showTrend(scores=selectedFocusedScore,name=selectedFolder+' focused')
	if save:
		plt.savefig('./plots/histograms/'+selectedFolder+'-'+fileName+'_trend.png', bbox_inches="tight")

	# loss comparison
	plt.figure(figsize=(20, 15))
	paper_fig_settings()
	lossComparison(name=selectedFolder+' dense',results=selectedDenseLoss)
	lossComparison(name=selectedFolder+' focused',results=selectedFocusedLoss)
	print(f'Area Under dense Curve',trapz(selectedDenseLoss))
	print(f'Area Under focused Curve',trapz(selectedFocusedLoss))
	if save:
		plt.savefig('./plots/histograms/'+selectedFolder+'-'+fileName+'_loss.png', bbox_inches="tight")

	# t-test will applied according to our hypotesis
	# 	we will make an assumption according to area under curve
	# 	and will say like this focused nn learn to solve
	# 	the problem more fast, accurate and efficient
	statistic, pvalue = ttest_ind(selectedDenseScore,selectedFocusedScore)
	print('statistic: ',float(statistic))
	print('pvalue: ',float(pvalue))

	plt.show()


if __name__ == "__main__":
	basePath = "./results"
	color = {'dense':		'royalblue'
			,'focused':		'seagreen'
			,'focused-025':	'seagreen'
			,'focused-02':	'mediumpurple'
			,'focused-01':	'orchid'
			,'focused-005':	'darkkhaki'}

	data={}
	loss = {}
	selectedDenseScore = np.array([])
	selectedFocusedScore = np.array([])
	selectedDenseLoss = np.array([])
	selectedFocusedLoss = np.array([])

	# np.seterr(divide='ignore', invalid='ignore')

	# # histogram display over score table on a problem
	bringData(folder=returnList(joinPath([basePath, 'test']))[[1]]
			,scorePosition=2 # index in the file list
			,lossPosition=0 # index in the file list
			,dataFolder='test'
			,pickle=True
			,save=True)