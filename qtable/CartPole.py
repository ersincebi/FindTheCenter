import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

learning_rate = 1e-3

env = gym.make('CartPole-v1')
env.reset()
goal_step = 500
score_requerinment = 50
initial_games = 10000

def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observersation = []
		for _ in range(goal_step):
			# env.render()
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(prev_observersation) > 0:
				game_memory.append([prev_observersation, action])

			prev_observersation = observation
			score += reward
			if done:
				break

		if score >= score_requerinment:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]
					
				training_data.append([data[0],output])

		env.reset()
		scores.append(score)

	training_data_save = np.array(training_data)
	np.save('save.npy', training_data_save)

	print('Avarage accepted score: ', mean(accepted_scores))
	print('Median accepted score: ', median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data

def neural_network_model(input_size):
	network = input_data(shape=[None, input_size, 1], name='input')

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(network, tensorboard_dir='log')

	return model


def train_model(training_data, model=False):
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	Y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size=len(X[0]))

	model.fit({'input':X}, {'targets':Y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai')

	return model



training_data = initial_population()
model = train_model(training_data)


scores = []
choices = []

for each_game in range(10):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_step):
		env.render()
		if len(prev_obs) == 0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
		
		choices.append(action)

		new_observation, reward, done, info = env.step(action)

		prev_obs = new_observation

		game_memory.append([new_observation, action])

		score += reward

		if done:
			break

	scores.append(score)

print('Avarage Score : ', mean(scores))
print('Choice 1: {} Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))
