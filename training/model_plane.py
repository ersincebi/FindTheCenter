import numpy as np
from . import training_data as td
from libs import game_rules as gr
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

def model():
	return train_model(td.training_data())

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

	network = fully_connected(network, 4, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=gr.LEARNING_RATE, loss='softmax_categorical_crossentropy', name='targets')

	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def train_model(training_data):
	X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	Y = [i[1] for i in training_data]

	model = neural_network_model(input_size=len(X[0]))

	model.fit({'input':X}, {'targets':Y}, n_epoch=5, snapshot_step=40, show_metric=True, run_id='findthecenter', )

	return model
