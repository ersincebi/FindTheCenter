import numpy as np
import matplotlib.pyplot as plt

mean = lambda lst, key: lst.count(key)/len(lst)

def plotTheValues(scores, name):
	x = []
	y = []
	for i in range(0, len(scores)):
		x.append(int(i))
		y.append(int(scores[i]))

	plt.figure(figsize=(20, 15))

	plt.title(name+'_score_grahp')

	plt.plot(x, y, label="score per run")

	plt.plot(x[-100:], [np.mean(y[-100:])] * len(y[-100:]), linestyle="--", label="last 100 runs average")

	trend_x = x[1:]
	z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
	p = np.poly1d(z)
	plt.plot(trend_x, p(trend_x), linestyle="-.", label="trend")


	plt.xlabel("episodes")
	plt.ylabel("scores")
	plt.legend(loc="upper left")
	
	plt.show()
	
	plt.savefig('./data/graphs/'+name+'_score_grahp.png', bbox_inches="tight")

def statistic(scores, choices):
	choice = set(choices)
	print()
	print('Average Score: ',sum(scores)/len(scores))
	for i in choice:
		print(f"choice {i}:{mean(choices,i)}")

def lossAccComparison(dense, focused, name):
	plt.figure(figsize=(16,8))
	plt.title(name+'_loss_grahp')
	plt.plot(dense['loss'])
	plt.plot(focused['loss'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['dense','focused'])
	plt.savefig('./data/graphs/'+name+'_loss_grahp.png', bbox_inches="tight")
	
	plt.figure(figsize=(16,8))
	plt.title(name+'_accuracy_grahp')
	plt.plot(dense['acc'])
	plt.plot(focused['acc'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['dense','focused'])
	plt.savefig('./data/graphs/'+name+'_accuracy_grahp.png', bbox_inches="tight")