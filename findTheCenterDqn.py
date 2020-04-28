import pyglet
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from libs.game_rules import TITLE, SIZE, LOW, BLACK_ARC_DIAMETER, RED_ARC_DIAMETER, VELOCITY, PENALTY, REWARD, MOVE_PENALTY, MAX_MOVE, HM_EPISODES, MAX_POS, ACTION_SPACE
from pyglet.sprite import Sprite
from libs.gameObjects import gameObjects, preload_image
from random import randrange
from math import floor

class findTheCenterDqn(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_location(400,100)
		self.frame_rate = 1/200.0

		self.episodes = 0
		self.move_count = 0

		self.reward = 0
		self.episode_reward = 0
		self.episode_rewards = np.array()

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

		self.prev_obs = np.array()
		self.current_state = [self.redArc-self.blackArc]

		self.model = model()

	def rePosition(self):
		self.redArc.posx = randrange(LOW, MAX_POS, VELOCITY)
		self.redArc.posy = randrange(LOW, MAX_POS, VELOCITY)
		self.current_state = [self.redArc-self.blackArc]
	
	def restart(self):
		self.rePosition()
		self.prev_obs = np.array()
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

		self.current_state = [self.redArc-self.blackArc]

		# collusion detection
		if self.redArc.posy == SIZE or self.redArc.posx == SIZE or self.redArc.posy == LOW and self.redArc.posx == LOW:
			self.reward = -PENALTY
		if self.redArc.posy>self.coly and self.redArc.posy<self.colx and self.redArc.posx>self.coly and self.redArc.posx<self.colx:
			# print(f'on #{self.episodes}, epsilon: {self.epsilon}, episode mean: {np.mean(self.episode_rewards[-SIZE:])}')
			print(f"succeed on episode #{self.episodes}")
			self.reward = REWARD
			
			self.makeNewPrediction()
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

	def makeNewPrediction(self):
		if len(self.prev_obs)!=0:
			self.prev_obs = np.array(self.prev_obs)
			action = np.argmax(self.model.predict(self.prev_obs.reshape(-1, len(self.prev_obs), 1))[0])
		else:
			action = np.random.randint(0, ACTION_SPACE)

		self.prev_obs = self.current_state

		self.choice = action

	def update(self, dt):
		self.move_count += 1
		if self.move_count != MAX_MOVE:
			# if agent finishes the episode
			self.makeNewPrediction()
			self.action(self.choice, dt)

		else:
			# when episode finished
			self.episode_rewards.append(self.episode_reward)
			self.makeNewPrediction()
			self.action(self.choice, dt)
			self.restart()

		if self.episodes == HM_EPISODES:
			# when all episodes are finished
			moving_avg = np.convolve(self.episode_rewards, np.ones((SIZE,)) / SIZE, mode='valid')

			plt.plot([i for i in range(len(moving_avg))], moving_avg)
			plt.xlabel(f'reward {SIZE} moving avarage')
			plt.ylabel('episode #')
			plt.show()


if __name__ == '__main__':
	window = findTheCenterDqn(SIZE, SIZE, TITLE, resizable = False)
	pyglet.clock.schedule_interval(window.update, window.frame_rate)
	pyglet.app.run()
