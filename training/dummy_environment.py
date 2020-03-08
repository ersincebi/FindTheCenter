from libs import game_rules as gr
from random import randrange
from math import floor

SIZE = 350
LOW = 0

RED_ARC_DIAMETER = 20
BLACK_ARC_DIAMETER = 65

MAX_POS = SIZE - RED_ARC_DIAMETER

class dummy_environment:
	def __init__(self):
		self.redArcState_x = 0
		self.redArcState_y = 0
		self.colx = floor((SIZE/2)+(BLACK_ARC_DIAMETER/2))
		self.coly = floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))
		self.done = False
		self.episode_reward = 0

	def rePosition(self):
		self.redArcState_x = randrange(LOW, MAX_POS, gr.VELOCITY)
		self.redArcState_y = randrange(LOW, MAX_POS, gr.VELOCITY)
		self.current_state = [self.redArcState_x,self.redArcState_y]

	def move(self, x, y):
		self.redArcState_x += x
		self.redArcState_y += y

		if self.redArcState_x < LOW:
			self.redArcState_x = LOW
		elif self.redArcState_x > SIZE-RED_ARC_DIAMETER:
			self.redArcState_x = SIZE-RED_ARC_DIAMETER

		if self.redArcState_y < LOW:
			self.redArcState_y = LOW
		elif self.redArcState_y > SIZE-RED_ARC_DIAMETER:
			self.redArcState_y = SIZE-RED_ARC_DIAMETER

		self.current_state = [self.redArcState_x,self.redArcState_y]

		# collusion detection
		if self.redArcState_y == SIZE or self.redArcState_x == SIZE or self.redArcState_y == LOW and self.redArcState_x == LOW:
			reward = -gr.PENALTY
		if self.redArcState_y>self.coly and self.redArcState_y<self.colx and self.redArcState_x>self.coly and self.redArcState_x<self.colx:
			reward = gr.REWARD
			self.done = True
		else:
			reward = -gr.MOVE_PENALTY

		self.episode_reward += reward


	def step(self, choice):
		if choice == 0:
			self.move(x=gr.VELOCITY, y=gr.VELOCITY)
		elif choice == 1:
			self.move(x=-gr.VELOCITY, y=-gr.VELOCITY)
		elif choice == 2:
			self.move(x=-gr.VELOCITY, y=gr.VELOCITY)
		elif choice == 3:
			self.move(x=gr.VELOCITY, y=-gr.VELOCITY)

		return self.current_state, self.episode_reward, self.done

	def reset(self):
		self.rePosition()
		self.episode_reward = 0
		self.done = False