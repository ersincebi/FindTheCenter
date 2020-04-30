import numpy as np
from . import dummy_environment as env
from libs.model import build_model, play_one_step, training_step
from libs.game_rules import HM_EPISODES, MAX_MOVE, ACTION_SPACE, EPSILON
from data.plottingData import statistic, plotTheValues
'''
from libs.Kfocusing import FocusedLayer1D
from libs.keras_utils import SGDwithLR, RMSpropwithClip,AdamwithClip
from keras.optimizers import RMSprop
from keras import backend as K
'''

model, optimizer = build_model(mode='focused')
env = env.dummy_environment()

def train_env():
	scores = []
	choices = []
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
		print(f"\rEpisode: {episode+1}, Steps: {step+1}, eps: {episilon}", end="")
		if episode > 50:
			training_step(model, optimizer)

	model.set_weights(best_weights)
	
	return scores, choices


def main():
	scores, choices = train_env()

	statistic(scores=scores, choices=choices)
	plotTheValues(scores=scores, name="findTheCenterDQN-3")