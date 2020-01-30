import pyglet
import numpy as np
from pyglet.sprite import Sprite
from gameObjects import gameObjects, preload_image
from random import randint

class findTheCenter(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_location(400,100)
		self.frame_rate = 1/60.0

		velocity = np.arange(10,500,20)
		
		self.velocityx = velocity[randint(0,24)]
		self.velocityy = velocity[randint(0,24)]
		
		self.main_batch = pyglet.graphics.Batch()
		backGround = pyglet.graphics.OrderedGroup(0)
		foreGround = pyglet.graphics.OrderedGroup(1)

		backGround_sprite = Sprite(preload_image('backGround.png'), batch=self.main_batch, group=backGround)
		self.backGround = gameObjects(0,0,backGround_sprite)

		blackArc_sprite = Sprite(preload_image('blackArc.png'), batch=self.main_batch, group=foreGround)
		self.blackArc = gameObjects(216,216,blackArc_sprite)

		redArc_sprite = Sprite(preload_image('redArc.png'), batch=self.main_batch, group=foreGround)
		self.redArc = gameObjects(0,0,redArc_sprite)
		

	def on_draw(self):
		self.clear()
		self.main_batch.draw()

	def move(self, dt):
		self.redArc.update()
		if self.redArc.posy < 500 - self.redArc.width and self.redArc.posx < 500 - self.redArc.width:
			self.redArc.posx += self.velocityx * dt
			self.redArc.posy += self.velocityy * dt

		# collusion detection
		if self.redArc.posy>220 and self.redArc.posy<250 and self.redArc.posx>220 and self.redArc.posx<250:
			self.velocityx = 0
			self.velocityy = 0

	def update(self, dt):
		self.move(dt)


width = 500
height = 500
title = "Find The Center"
if __name__ == '__main__':
	window = findTheCenter(width,height,title, resizable = False)
	pyglet.clock.schedule_interval(window.update, window.frame_rate)
	pyglet.app.run()
