import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
from libs.game_rules import ACTION_SPACE, OBSERVATION_SPACE, LEARNING_RATE, BATCH_SIZE, DISCOUNT



replay_memory = deque(maxlen=2000)
loss_fn = keras.losses.mean_squared_error

def build_model(N=32
				,mode='dense'
				,optimizer_s='adam'):

	keras.backend.clear_session()

	if optimizer_s == 'SGDwithLR':
		pass # optimizer = SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict)

	elif optimizer_s == 'AdamwithCli':
		pass # optimizer = AdamwithClip()

	elif optimizer_s=='RMSpropwithClip':
		pass # optimizer = RMSpropwithClip(lr=0.001, rho=0.9, epsilon=None, decay=0.0,clips=clip_dict)

	elif optimizer_s=='adam':
		optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)

	else:
		pass # optimizer = SGD(lr=0.01, momentum=0.9)


	if mode=='dense':
		layer = keras.layers.Dense(32, activation="elu")

	elif mode=='focused':
		pass
		# layer = FocusedLayer1D(units=N
		# 					,name='focus-1'
		# 					,activation='relu'
		# 					,init_sigma=0.25
		# 					,init_mu='spread'
		# 					,init_w= None
		# 					,train_sigma=True
		# 					,train_weights=True
		# 					,train_mu = True
		# 					,normed=2)

	model = keras.models.Sequential([
		keras.layers.Dense(32, activation="elu", input_shape=OBSERVATION_SPACE),
		layer,
		keras.layers.Dense(ACTION_SPACE)
	])

	print(model.summary())

	return model, optimizer

def sample_experiences():
	indices = np.random.randint(len(replay_memory), size=BATCH_SIZE)
	batch = [replay_memory[index] for index in indices]
	states, actions, rewards, next_states, dones = [
		np.array([experience[field_index] for experience in batch])
		for field_index in range(5)]
	return states, actions, rewards, next_states, dones

def training_step(model, optimizer):
	states, actions, rewards, next_states, dones = sample_experiences()
	next_Q_values = model.predict(next_states)
	max_next_Q_values = np.max(next_Q_values, axis=1)
	target_Q_values = (rewards +
					(1 - dones) * DISCOUNT * max_next_Q_values)
	target_Q_values = target_Q_values.reshape(-1, 1)
	mask = tf.one_hot(actions, ACTION_SPACE)
	with tf.GradientTape() as tape:
		all_Q_values = model(states)
		Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
		loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))

def play_one_step(env, state, epsilon, model):
	action = epsilon_greedy_policy(model, state, epsilon)
	next_state, reward, done = env.step(action)
	replay_memory.append((state, action, reward, next_state, done))
	return next_state, reward, done, action

def epsilon_greedy_policy(model, state, epsilon=0):
	if np.random.rand() < epsilon:
		return np.random.randint(2)
	else:
		state = np.array(state)
		Q_values = model.predict(state[np.newaxis])
		return np.argmax(Q_values[0])