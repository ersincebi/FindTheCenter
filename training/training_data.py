import numpy as np
from . import dummy_environment as env
from libs import game_rules as gr
from random import randrange
from statistics import mean, median
from collections import Counter

env = env.dummy_environment()
env.reset()

def training_data():
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(gr.HM_EPISODES):
		score = 0
		game_memory = []
		prev_observersation = []
		for _ in range(gr.MAX_MOVE):
			action = randrange(0,4)
			observation, reward, done = env.step(action)

			if len(prev_observersation) > 0:
				game_memory.append([prev_observersation, action])

			prev_observersation = observation
			score += reward

			if done:
				break


		if score >= 10:
			accepted_scores.append(score)
			for data in game_memory:
				output = [0,0,0,0]
				output[data[1]] = 1
				training_data.append([data[0],output])

		env.reset()
		scores.append(score)

	# np.save('save.npy', np.array(training_data))

	print('Avarage accepted score: ', mean(accepted_scores))
	print('Median accepted score: ', median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data