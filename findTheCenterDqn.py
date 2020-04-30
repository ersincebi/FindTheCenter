import pyglet
import numpy as np
from libs.game_rules import TITLE, SIZE, LOW, BLACK_ARC_DIAMETER, RED_ARC_DIAMETER, VELOCITY, PENALTY, REWARD, MOVE_PENALTY, MAX_MOVE, HM_EPISODES, MAX_POS, ACTION_SPACE, EPSILON
from pyglet.sprite import Sprite
from libs.gameObjects import gameObjects, preload_image
from random import randrange
from data.plottingData import statistic, plotTheValues
from libs.model import build_model, training_step, epsilon_greedy_policy, REPLAY_MEMORY

class findTheCenterDqn(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_location(400,100)
		self.frame_rate = 1/200.0

		self.episodes = 0
		self.move_count = 0

		self.score = 0
		self.episode_score = 0
		self.best_score = 0
		self.done = False
		self.episode_scores = []
		self.episode_choices = []

		self.epsilon = EPSILON(self.episodes)

		self.main_batch = pyglet.graphics.Batch()
		backGround = pyglet.graphics.OrderedGroup(0)
		foreGround = pyglet.graphics.OrderedGroup(1)

		backGround_sprite = Sprite(preload_image('backGround.png'), batch=self.main_batch, group=backGround)
		self.backGround = gameObjects(LOW,LOW,backGround_sprite)

		pos = np.floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))

		self.colx = np.floor((SIZE/2)+(BLACK_ARC_DIAMETER/2))
		self.coly = np.floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))

		blackArc_sprite = Sprite(preload_image('blackArc.png'), batch=self.main_batch, group=foreGround)
		self.blackArc = gameObjects(pos,pos,blackArc_sprite)

		redArc_sprite = Sprite(preload_image('redArc.png'), batch=self.main_batch, group=foreGround)
		self.redArc = gameObjects(LOW,LOW,redArc_sprite)

		self.prev_obs = np.array([])
		self.current_state = np.array(self.redArc-self.blackArc)

		self.model, self.optimizer = build_model()

	def rePosition(self):
		self.redArc.posx = randrange(LOW, MAX_POS, VELOCITY)
		self.redArc.posy = randrange(LOW, MAX_POS, VELOCITY)
		self.current_state = np.array(self.redArc-self.blackArc)
	
	def restart(self):
		self.rePosition()
		self.prev_obs = np.array([])
		self.move_count = 0
		self.episode_score = 0
		self.done = False
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

		self.current_state = np.array(self.redArc-self.blackArc)

		# collusion detection
		if self.redArc.posy == SIZE or self.redArc.posx == SIZE or self.redArc.posy == LOW and self.redArc.posx == LOW:
			self.score = -PENALTY
		if self.redArc.posy>self.coly and self.redArc.posy<self.colx and self.redArc.posx>self.coly and self.redArc.posx<self.colx:
			self.score = REWARD
			self.done = True
			
			self.makeNewPrediction()
			self.restart()
			
		else:
			self.score = -MOVE_PENALTY

		self.episode_score += self.score

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
		self.choice = epsilon_greedy_policy(self.model, self.current_state, self.epsilon)

		REPLAY_MEMORY.append((self.prev_obs, self.choice, self.score, self.current_state, self.done))

		self.prev_obs = self.current_state

		self.episode_choices.append(self.choice)

	def update(self, dt):
		self.move_count += 1
		if self.move_count != MAX_MOVE:
			# when agent in the episode
			self.epsilon = EPSILON(self.episodes)

			self.makeNewPrediction()
			self.action(self.choice, dt)

		else:
			# when episode finished
			print(f"\rEpisode: {self.episodes+1}, Steps: {self.move_count+1}, eps: {self.epsilon}", end="")
			self.episode_scores.append(self.episode_score)
			self.makeNewPrediction()
			self.action(self.choice, dt)
			self.restart()

		if self.episodes == HM_EPISODES:
			# when all episodes are finished
			statistic(scores=self.episode_scores, choices=self.episode_choices)
			plotTheValues(scores=self.episode_scores, name="findTheCenterDQN-1")


if __name__ == '__main__':
	window = findTheCenterDqn(SIZE, SIZE, TITLE, resizable = False)
	pyglet.clock.schedule_interval(window.update, window.frame_rate)
	pyglet.app.run()
