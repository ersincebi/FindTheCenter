import pyglet
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from pyglet.sprite import Sprite
from gameObjects import gameObjects, preload_image
from random import randint
from math import floor

SIZE = 500
TITLE = "Find The Center"


# CONSTANT VARIABLES
########################################################
QTABLESIZE = 10

HM_EPISODES = 25000 # how many episodes
MOVE_PENALTY = 1
MAX_MOVE = 200
PENALTY = 300
REWARD = 25

#how higher is epsilon, agent makes more random action
EPSILON = 0.0
EPS_DECAY = 0.9998

LEARNING_RATE = 0.1
DISCOUNT = 0.95

VELOCITY = 50
########################################################
# q_table initialization
q_table = {}
for x in range(-SIZE+1, SIZE):
	for y in range(-SIZE+1, SIZE):
		q_table[(x,y)] = [np.random.uniform(-5,0) for _ in range(4)]
########################################################

class findTheCenter(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_location(400,100)
		self.frame_rate = 1/60.0

		self.episodes = 0
		self.move_count = 0

		self.epsilon = EPSILON

		self.episode_reward = 0
		self.episode_rewards = []

		self.main_batch = pyglet.graphics.Batch()
		backGround = pyglet.graphics.OrderedGroup(0)
		foreGround = pyglet.graphics.OrderedGroup(1)

		backGround_sprite = Sprite(preload_image('backGround.png'), batch=self.main_batch, group=backGround)
		self.backGround = gameObjects(0,0,backGround_sprite)

		blackArc_sprite = Sprite(preload_image('blackArc.png'), batch=self.main_batch, group=foreGround)
		self.blackArc = gameObjects(216,216,blackArc_sprite)

		redArc_sprite = Sprite(preload_image('redArc.png'), batch=self.main_batch, group=foreGround)
		self.redArc = gameObjects(0,0,redArc_sprite)

	def rePosition(self):
		self.redArc.posx = randint(0,460)
		self.redArc.posy = randint(0,460)

	def restart(self):
		self.rePosition()
		self.epsilon *= EPS_DECAY
		self.move_count = 0
		self.episode_reward = 0
		self.episodes += 1

	def on_draw(self):
		self.clear()
		self.main_batch.draw()

	def move(self, dt, x=False, y=False):
		self.redArc.update()
		if not x:
			self.redArc.posx += np.random.randint(-1,2)
		else:
			self.redArc.posx += x

		if not y:
			self.redArc.posy += np.random.randint(-1,2)
		else:
			self.redArc.posy += y

		if self.redArc.posx < 0:
			self.redArc.posx = 0
		elif self.redArc.posx > SIZE-40:
			self.redArc.posx = SIZE-40


		if self.redArc.posy < 0:
			self.redArc.posy = 0
		elif self.redArc.posy > SIZE-40:
			self.redArc.posy = SIZE-40

		# collusion detection
		if self.redArc.posy < 500 - self.redArc.width and self.redArc.posx < 500 - self.redArc.width and self.redArc.posy > 0 and self.redArc.posx > 0:
			self.reward = -MOVE_PENALTY
		elif self.redArc.posy>200 and self.redArc.posy<265 and self.redArc.posx>200 and self.redArc.posx<265:
			print(f'on #{self.episodes}, epsilon: {self.epsilon}')
			self.reward = REWARD
		else:
			self.reward = -PENALTY

		self.episode_reward += self.reward

	def action(self, choice, dt):
		if choice == 0:
			self.move(dt, x=VELOCITY, y=VELOCITY)
		elif choice == 1:
			self.move(dt, x=-VELOCITY, y=-VELOCITY)
		elif choice == 2:
			self.move(dt, x=-VELOCITY, y=VELOCITY)
		elif choice == 3:
			self.move(dt, x=VELOCITY, y=-VELOCITY)

	def makeNewObservation(self, dt):
		self.observation = (self.redArc-self.blackArc)

		if np.random.random() > self.epsilon:
			action = np.argmax(q_table[self.observation])
		else:
			action = np.random.randint(0,3)

		self.action(action, dt)

		# new prediction
		max_future_q = np.max(q_table[self.observation])
		current_q = q_table[self.observation][action]

		if self.reward == REWARD:
			new_q = REWARD
		elif self.reward == -PENALTY:
			new_q = -PENALTY
		else:
			# q_table optimization formula
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (self.reward + DISCOUNT * max_future_q)

		q_table[self.observation][action] = new_q

	def update(self, dt):
		self.move_count += 1
		if self.move_count != MAX_MOVE:
			# if agent finishes the episode
			self.makeNewObservation(dt)

		elif self.episodes == HM_EPISODES:
			# when all episodes are finished
			moving_avg = np.convolve(episode_rewards, np.ones((SIZE,)) / SIZE, mode='valid')

			plt.plot([i for i in range(len(moving_avg))], moving_avg)
			plt.ylabel(f'reward {SIZE} moving avarage')
			plt.ylabel('episode #')
			plt.show()

			with open(f'qtable-{int(time.time())}.pickle', 'wb') as f:
				pickle.dump(q_table, f)
			
		else:
			# when episode finished
			self.episode_rewards.append(self.episode_reward)
			self.makeNewObservation(dt)
			self.restart()

if __name__ == '__main__':
	window = findTheCenter(SIZE, SIZE, TITLE, resizable = False)
	pyglet.clock.schedule_interval(window.update, window.frame_rate)
	pyglet.app.run()
