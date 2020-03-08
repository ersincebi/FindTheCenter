import pyglet
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import libs.game_rules as gr
from pyglet.sprite import Sprite
from libs.gameObjects import gameObjects, preload_image
from random import randrange
from math import floor

SIZE = 350
LOW = 0
TITLE = "Find The Center"

# CONSTANT VARIABLES
########################################################
RED_ARC_DIAMETER = 20
BLACK_ARC_DIAMETER = 65
########################################################
# q_table initialization
initial_q_table = 'qtable-1582528297.pickle'
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
		self.frame_rate = 1/60.0

		self.episodes = 0
		self.move_count = 0

		self.epsilon = gr.EPSILON

		self.reward = 0
		self.episode_reward = 0
		self.episode_rewards = []

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
		max_pos = SIZE - RED_ARC_DIAMETER
		self.redArc.posx = randrange(LOW, max_pos, gr.VELOCITY)
		self.redArc.posy = randrange(LOW, max_pos, gr.VELOCITY)
	
	def restart(self):
		self.rePosition()
		self.epsilon *= gr.EPS_DECAY
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

		if self.redArc.posx < LOW:
			self.redArc.posx = LOW
		elif self.redArc.posx > SIZE-RED_ARC_DIAMETER:
			self.redArc.posx = SIZE-RED_ARC_DIAMETER


		if self.redArc.posy < LOW:
			self.redArc.posy = LOW
		elif self.redArc.posy > SIZE-RED_ARC_DIAMETER:
			self.redArc.posy = SIZE-RED_ARC_DIAMETER

		# collusion detection
		colx = floor((SIZE/2)+(BLACK_ARC_DIAMETER/2))
		coly = floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))

		if self.redArc.posy == SIZE or self.redArc.posx == SIZE or self.redArc.posy == LOW and self.redArc.posx == LOW:
			self.reward = -gr.PENALTY
		if self.redArc.posy>coly and self.redArc.posy<colx and self.redArc.posx>coly and self.redArc.posx<colx:
			# print(f'on #{self.episodes}, epsilon: {self.epsilon}, episode mean: {np.mean(self.episode_rewards[-SIZE:])}')
			print(f"succeed on episode #{self.episodes}, epsilon: {self.epsilon}")
			self.reward = gr.REWARD
			
			self.makeNewObservation()
			self.restart()
			
		else:
			self.reward = -gr.MOVE_PENALTY

		self.episode_reward += self.reward

	def action(self, choice, dt):
		if choice == 0:
			self.move(dt, x=gr.VELOCITY, y=gr.VELOCITY)
		elif choice == 1:
			self.move(dt, x=-gr.VELOCITY, y=-gr.VELOCITY)
		elif choice == 2:
			self.move(dt, x=-gr.VELOCITY, y=gr.VELOCITY)
		elif choice == 3:
			self.move(dt, x=gr.VELOCITY, y=-gr.VELOCITY)

	def makeNewObservation(self):
		self.observation = (self.redArc-self.blackArc)

		if np.random.random() > self.epsilon:
			action = np.argmax(q_table[self.observation])
		else:
			action = np.random.randint(0,3)

		# new prediction
		max_future_q = np.max(q_table[self.observation])
		current_q = q_table[self.observation][action]

		if self.reward == gr.REWARD:
			new_q = gr.REWARD
		elif self.reward == -gr.PENALTY:
			new_q = -gr.PENALTY
		else:
			# q_table optimization formula
			new_q = (1 - gr.LEARNING_RATE) * current_q + gr.LEARNING_RATE * (self.reward + gr.DISCOUNT * max_future_q)

		q_table[self.observation][action] = new_q

		self.choice = action

	def update(self, dt):
		self.move_count += 1
		if self.move_count != gr.MAX_MOVE:
			# if agent finishes the episode
			self.makeNewObservation()
			self.action(self.choice, dt)

		else:
			# when episode finished
			self.episode_rewards.append(self.episode_reward)
			self.makeNewObservation()
			self.action(self.choice, dt)
			self.restart()

		if self.episodes == gr.HM_EPISODES:
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
