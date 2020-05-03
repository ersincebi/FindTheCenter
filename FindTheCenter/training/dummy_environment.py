from libs.game_rules import VELOCITY, PENALTY, REWARD, MOVE_PENALTY, RED_ARC_DIAMETER, BLACK_ARC_DIAMETER, LOW, SIZE, MAX_POS
from random import randrange
from math import floor

class dummy_environment:
	def __init__(self):
		self.redArcState_x = 0
		self.redArcState_y = 0
		self.colx = floor((SIZE/2)+(BLACK_ARC_DIAMETER/2))
		self.coly = floor((SIZE/2)-(BLACK_ARC_DIAMETER/2))
		self.done = False
		self.episode_reward = 0

	def rePosition(self):
		self.redArcState_x = randrange(LOW, MAX_POS, VELOCITY)
		self.redArcState_y = randrange(LOW, MAX_POS, VELOCITY)
		self.current_state = [self.redArcState_x-self.colx, self.redArcState_y-self.coly]

	def move(self, x, y):
		self.redArcState_x += x
		self.redArcState_y += y

		if self.redArcState_x < LOW:
			self.redArcState_x = LOW
		elif self.redArcState_x > MAX_POS:
			self.redArcState_x = MAX_POS

		if self.redArcState_y < LOW:
			self.redArcState_y = LOW
		elif self.redArcState_y > MAX_POS:
			self.redArcState_y = MAX_POS

		self.current_state = [self.redArcState_x-self.colx, self.redArcState_y-self.coly]

		# collusion detection
		if self.redArcState_y == SIZE or self.redArcState_x == SIZE or self.redArcState_y == LOW and self.redArcState_x == LOW:
			reward = -PENALTY
		if self.redArcState_y>self.coly and self.redArcState_y<self.colx and self.redArcState_x>self.coly and self.redArcState_x<self.colx:
			reward = REWARD
			self.done = True
		else:
			reward = -MOVE_PENALTY

		self.episode_reward += reward


	def step(self, choice):
		if choice == 0:
			self.move(x=VELOCITY, y=VELOCITY)
		elif choice == 1:
			self.move(x=-VELOCITY, y=-VELOCITY)
		elif choice == 2:
			self.move(x=-VELOCITY, y=VELOCITY)
		elif choice == 3:
			self.move(x=VELOCITY, y=-VELOCITY)

		return self.current_state, self.episode_reward, self.done

	def reset(self):
		self.rePosition()
		self.episode_reward = 0
		self.done = False
		return self.current_state