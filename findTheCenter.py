import pyglet
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from libs.game_rules import TITLE, SIZE, LOW, HM_EPISODES, EPS_DECAY, BLACK_ARC_DIAMETER, RED_ARC_DIAMETER, EPSILON, MAX_MOVE, VELOCITY, PENALTY, MOVE_PENALTY, REWARD, LEARNING_RATE, DISCOUNT, MAX_POS, ACTION_SPACE
from pyglet.sprite import Sprite
from libs.gameObjects import gameObjects, preload_image
from random import randrange
from math import floor

# q_table initialization
initial_q_table = None
if initial_q_table is None:
	q_table = {}
	for x in range(-SIZE+1, SIZE):
		for y in range(-SIZE+1, SIZE):
			q_table[(x,y)] = [np.random.uniform(-5,0) for _ in range(4)]
else:
	with open(initial_q_table, 'rb') as f:
		q_table = pickle.load(f)
########################################################

class findTheCenter(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_location(400,100)
		self.frame_rate = 1/200.0

		self.episodes = 0
		self.move_count = 0

		self.epsilon = EPSILON

		self.reward = 0
		self.episode_reward = 0
		self.episode_rewards = np.array()

		self.main_batch = pyglet.graphics.Batch()
		backGround = pyglet.graphics.OrderedGroup(0)
		foreGround = pyglet.graphics.OrderedGroup(1)

		backGround_sprite = Sprite(preload_image('backGround.png'), batch=self.main_batch, group=backGround)
		self.backGround = gameObjects(LOW,LOW,backGround_sprite)

		pos = floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))
		blackArc_sprite = Sprite(preload_image('blackArc.png'), batch=self.main_batch, group=foreGround)
		self.blackArc = gameObjects(pos,pos,blackArc_sprite)

		redArc_sprite = Sprite(preload_image('redArc.png'), batch=self.main_batch, group=foreGround)
		self.redArc = gameObjects(LOW,LOW,redArc_sprite)

	def rePosition(self):
		self.redArc.posx = randrange(LOW, MAX_POS, VELOCITY)
		self.redArc.posy = randrange(LOW, MAX_POS, VELOCITY)
	
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
		
		self.redArc.posx += x
		self.redArc.posy += y

		if self.redArc.posx < LOW:
			self.redArc.posx = LOW
		elif self.redArc.posx > MAX_POS:
			self.redArc.posx = MAX_POS


		if self.redArc.posy < LOW:
			self.redArc.posy = LOW
		elif self.redArc.posy > MAX_POS:
			self.redArc.posy = MAX_POS

		# collusion detection
		colx = floor((SIZE/2)+(BLACK_ARC_DIAMETER/2))
		coly = floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))

		if self.redArc.posy == SIZE or self.redArc.posx == SIZE or self.redArc.posy == LOW and self.redArc.posx == LOW:
			self.reward = -PENALTY
		if self.redArc.posy>coly and self.redArc.posy<colx and self.redArc.posx>coly and self.redArc.posx<colx:
			# print(f'on #{self.episodes}, epsilon: {self.epsilon}, episode mean: {np.mean(self.episode_rewards[-SIZE:])}')
			print(f"succeed on episode #{self.episodes}, epsilon: {self.epsilon}")
			self.reward = REWARD
			
			self.makeNewObservation()
			self.restart()
			
		else:
			self.reward = -MOVE_PENALTY

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

	def makeNewObservation(self):
		self.observation = (self.redArc-self.blackArc)

		if np.random.random() > self.epsilon:
			action = np.argmax(q_table[self.observation])
		else:
			action = np.random.randint(0, ACTION_SPACE)

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

		self.choice = action

	def update(self, dt):
		self.move_count += 1
		if self.move_count != MAX_MOVE:
			# if agent finishes the episode
			self.makeNewObservation()
			self.action(self.choice, dt)

		else:
			# when episode finished
			self.episode_rewards.append(self.episode_reward)
			self.makeNewObservation()
			self.action(self.choice, dt)
			self.restart()

		if self.episodes == HM_EPISODES:
			# when all episodes are finished
			moving_avg = np.convolve(self.episode_rewards, np.ones((SIZE,)) / SIZE, mode='valid')

			plt.plot([i for i in range(len(moving_avg))], moving_avg)
			plt.xlabel(f'reward {SIZE} moving avarage')
			plt.ylabel('episode #')
			plt.show()

			with open(f'qtable-{int(time.time())}.pickle', 'wb') as f:
				pickle.dump(q_table, f)


if __name__ == '__main__':
	window = findTheCenter(SIZE, SIZE, TITLE, resizable = False)
	pyglet.clock.schedule_interval(window.update, window.frame_rate)
	pyglet.app.run()
