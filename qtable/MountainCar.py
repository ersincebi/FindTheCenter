import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000

SHOW_EVERY = 500

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#how higher is epsilon, agent makes more random action
epsilon = 0.5

START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 # // -> gives result of integer

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=DISCRETE_OS_SIZE + [env.action_space.n])

ep_rewards = []
aggr_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE
	return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
	episode_reward = 0
	if episode % SHOW_EVERY == 0:
		render = True
		print(episode)
	else:
		render = False
	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done:
		# 0 => left
		# 1 => do nothing
		# 2 => right
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		new_state, reward, done, _ = env.step(action)

		episode_reward += reward

		new_descrete_state = get_discrete_state(new_state)
		if render:
			env.render()

		if not done:
			max_future_q = np.max(q_table[new_descrete_state])
			current_q = q_table[discrete_state + (action, )]
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
			q_table[discrete_state + (action, )] = new_q
		elif new_state[0] >= env.goal_position:
			print("succeed on episode: ", episode)
			q_table[discrete_state + (action, )] = 0

		discrete_state = new_descrete_state

	if END_EPSILON_DECAYING >=episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

	ep_rewards.append(episode_reward)

	# if not episode % 10:
	# 	np.save(f'qtables/{episode}-qtable.npy', q_table)

	if not episode % SHOW_EVERY:
		avarage_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(avarage_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

		print(f'ep: {episode}, avg: {avarage_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}')

env.close()

plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label="avg rewards")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label="min rewards")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label="max rewards")
plt.legend(loc=4)
plt.show()