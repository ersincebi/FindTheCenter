import pyglet
from pyglet.sprite import Sprite
from gameObjects import gameObjects, preload_image
from random import randint

class findTheCenter(pyglet.window.Window):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_location(400,100)
		self.frame_rate = 1/60.0
		self.velocity = 50
		self.main_batch = pyglet.graphics.Batch()

		self.redArc_list = []
		self.redArcSprite = preload_image('redArc.png')
		for i in range(20):
			self.redArc_list.append(gameObjects(randint(0,450),randint(0,450),Sprite(self.redArcSprite, batch=self.main_batch)))
		
		# redArc_sprite = Sprite(preload_image('redArc.png'), batch=self.main_batch)
		# self.redArc = gameObjects(0,0,redArc_sprite)

		backGround_sprite = Sprite(preload_image('backGround.png'), batch=self.main_batch)
		self.backGround = gameObjects(0,0,backGround_sprite)
		
		blackArc_sprite = Sprite(preload_image('blackArc.png'), batch=self.main_batch)
		self.blackArc = gameObjects(216,216,blackArc_sprite)
		

	def on_draw(self):
		self.clear()
		self.main_batch.draw()

	def moveUp(self, dt):
		for redArc in self.redArc_list:
			redArc.update()
			if redArc.posy < 500 - redArc.width:
				redArc.posy += self.velocity * dt
				
		# self.redArc.update()
		# if self.redArc.posy < 500 - self.redArc.width:
		# 	self.redArc.posy += self.velocity * dt

	def update(self, dt):
		self.moveUp(dt)


width = 500
height = 500
title = "Find The Center"
if __name__ == '__main__':
	window = findTheCenter(width,height,title, resizable = False)
	pyglet.clock.schedule_interval(window.update, window.frame_rate)
	pyglet.app.run()
