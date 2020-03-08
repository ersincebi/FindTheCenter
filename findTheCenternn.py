import pyglet
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import libs.game_rules as gr
import training.model_plane as mp
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

class findTheCenter(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_location(400,100)
		self.frame_rate = 1/60.0

		self.episodes = 0
		self.move_count = 0

		self.reward = 0
		self.episode_reward = 0
		self.episode_rewards = []

		self.main_batch = pyglet.graphics.Batch()
		backGround = pyglet.graphics.OrderedGroup(0)
		foreGround = pyglet.graphics.OrderedGroup(1)

		backGround_sprite = Sprite(preload_image('backGround.png'), batch=self.main_batch, group=backGround)
		self.backGround = gameObjects(LOW,LOW,backGround_sprite)

		pos = floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))

		self.colx = floor((SIZE/2)+(BLACK_ARC_DIAMETER/2))
		self.coly = floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))

		blackArc_sprite = Sprite(preload_image('blackArc.png'), batch=self.main_batch, group=foreGround)
		self.blackArc = gameObjects(pos,pos,blackArc_sprite)

		redArc_sprite = Sprite(preload_image('redArc.png'), batch=self.main_batch, group=foreGround)
		self.redArc = gameObjects(LOW,LOW,redArc_sprite)

		self.prev_obs = []
		self.state = [0,1]
		self.current_state = [0,1]

		self.model = mp.model()

	def rePosition(self):
		max_pos = SIZE - RED_ARC_DIAMETER
		self.redArc.posx = randrange(LOW, max_pos, gr.VELOCITY)
		self.redArc.posy = randrange(LOW, max_pos, gr.VELOCITY)
		self.state = [self.redArc.posx,self.redArc.posy]
	
	def restart(self):
		self.rePosition()
		self.prev_obs = []
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
		elif self.redArc.posx > SIZE-RED_ARC_DIAMETER:
			self.redArc.posx = SIZE-RED_ARC_DIAMETER


		if self.redArc.posy < LOW:
			self.redArc.posy = LOW
		elif self.redArc.posy > SIZE-RED_ARC_DIAMETER:
			self.redArc.posy = SIZE-RED_ARC_DIAMETER

		self.current_state = [self.redArc.posx,self.redArc.posy]
		# collusion detection
		if self.redArc.posy == SIZE or self.redArc.posx == SIZE or self.redArc.posy == LOW and self.redArc.posx == LOW:
			self.reward = -gr.PENALTY
		if self.redArc.posy>self.coly and self.redArc.posy<self.colx and self.redArc.posx>self.coly and self.redArc.posx<self.colx:
			# print(f'on #{self.episodes}, epsilon: {self.epsilon}, episode mean: {np.mean(self.episode_rewards[-SIZE:])}')
			print(f"succeed on episode #{self.episodes}")
			self.reward = gr.REWARD
			
			self.makeNewPrediction()
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

	def makeNewPrediction(self):
		if len(self.prev_obs)!=0:
			self.prev_obs = np.array(self.prev_obs)
			action = np.argmax(self.model.predict(self.prev_obs.reshape(-1, len(self.prev_obs), 1))[0])
		else:
			action = np.random.randint(0,3)

		self.prev_obs = self.current_state

		self.choice = action

	def update(self, dt):
		self.move_count += 1
		if self.move_count != gr.MAX_MOVE:
			# if agent finishes the episode
			self.makeNewPrediction()
			self.action(self.choice, dt)

		else:
			# when episode finished
			self.episode_rewards.append(self.episode_reward)
			self.makeNewPrediction()
			self.action(self.choice, dt)
			self.restart()

		if self.episodes == gr.HM_EPISODES:
			# when all episodes are finished
			moving_avg = np.convolve(self.episode_rewards, np.ones((SIZE,)) / SIZE, mode='valid')

			plt.plot([i for i in range(len(moving_avg))], moving_avg)
			plt.xlabel(f'reward {SIZE} moving avarage')
			plt.ylabel('episode #')
			plt.show()


if __name__ == '__main__':
	window = findTheCenter(SIZE, SIZE, TITLE, resizable = False)
	pyglet.clock.schedule_interval(window.update, window.frame_rate)
	pyglet.app.run()
