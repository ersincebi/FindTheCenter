import numpy as np
from time import time
from . import dummy_environment as env
from libs.model import build_model, play_one_step, training_step, REPLAY_MEMORY
from libs.game_rules import HM_EPISODES, MAX_MOVE, ACTION_SPACE, EPSILON, BATCH_SIZE
from data.plottingData import statistic, plotTheValues, lossAccComparison
from tensorflow import keras

NAME = "findTheCenterDQN-{}".format(int(time()))

logDir="logs\\{}".format(NAME)

env = env.dummy_environment()

def train_env(mode):
	model, optimizer = build_model(mode=mode)
	scores = []
	choices = []
	metrics = []
	best_score = MAX_MOVE
	for episode in range(HM_EPISODES):
		score = 0
		obs = env.reset()
		for step in range(MAX_MOVE):
			episilon = EPSILON(episode)
			
			obs, reward, done, action = play_one_step(env, obs, episilon, model)
			
			choices.append(action)

			score += reward

			if done:
				break
				
		scores.append(score)
		if step < best_score:
			best_weights = model.get_weights()
			best_score = step
		if episode > 50:
			lossValue = training_step(model, optimizer, mode)
			metrics.append(lossValue)
	
		print(f"\rEpisode: {episode+1}, Steps: {step+1}, eps: {episilon}", end="")
	
	model.set_weights(best_weights)

	return scores, choices, metrics, model

def saveModel(mode, metrics, model, scores):
	model.save('./data/models/{}-{}-{}.h5'.format(NAME, mode, int(time())))
	path = './data/results/'
	metricsPath = path+'{}-{}-metrics-{}.npy'.format(NAME, mode, int(time()))
	np.save(metricsPath, np.array(metrics))
	scoresPath = path+'{}-{}-scores-{}.npy'.format(NAME, mode, int(time()))
	np.save(scoresPath, np.array(scores))
	replayPath = path+'{}-{}-replay-{}.npy'.format(NAME, mode, int(time()))
	np.save(replayPath, np.array(REPLAY_MEMORY))

def main():
	modes=['dense', 'focused']
	# Dense Section
	scores, choices, dense, model = train_env(mode=modes[0])
	statistic(scores=scores, choices=choices)
	plotTheValues(scores=scores, name='{}-{}'.format(NAME,modes[0]))
	
	saveModel(modes[0], dense, model, scores)
	
	# Focused Section
	scores, choices, focused, model = train_env(mode=modes[1])
	statistic(scores=scores, choices=choices)
	plotTheValues(scores=scores, name='{}-{}'.format(NAME,modes[1]))

	lossAccComparison(focused=focused, dense=dense, name=NAME)

	saveModel(modes[1], focused, model, scores)