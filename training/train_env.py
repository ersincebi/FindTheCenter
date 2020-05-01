import numpy as np
from . import dummy_environment as env
from libs.model import build_model, play_one_step, training_step
from libs.game_rules import HM_EPISODES, MAX_MOVE, ACTION_SPACE, EPSILON
from data.plottingData import statistic, plotTheValues, lossAccComparison

env = env.dummy_environment()

def train_env(mode):
	model, optimizer = build_model(mode=mode)
	scores = []
	choices = []
	metrics = {'loss':[], 'acc':[]}
	best_score = 0
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
		if step > best_score:
			best_weights = model.get_weights()
			best_score = step
		if episode > 50:
			lossValue, accurancyValue = training_step(model, optimizer)
			metrics['loss'].append(lossValue)
			metrics['acc'].append(accurancyValue)

		print(f"\rEpisode: {episode+1}, Steps: {step+1}, eps: {episilon}", end="")

	model.set_weights(best_weights)
	
	return scores, choices, metrics


def main():
	# Dense Section
	scores, choices, dense = train_env(mode='dense')
	statistic(scores=scores, choices=choices)
	plotTheValues(scores=scores, name='findTheCenterDQN-dense-4')
	
	# Focused Section
	scores, choices, focused = train_env(mode='focused')
	statistic(scores=scores, choices=choices)
	plotTheValues(scores=scores, name='findTheCenterDQN-focused-4')

	lossAccComparison(focused=focused, dense=dense, name='findTheCenterDQN-4')