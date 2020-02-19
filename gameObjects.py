import pyglet

def preload_image(image):
	return pyglet.image.load('shapes/' + image)

class gameObjects:
	def __init__(self, posx, posy, sprite=None):
		self.posx = posx
		self.posy = posy
		if sprite is not None:
			self.sprite = sprite
			self.sprite.x = self.posx
			self.sprite.y = self.posy
			self.width = self.sprite.width
			self.height = self.sprite.height

	def __str__(self):
		return f'{self.sprite.x},{self.sprite.y}'

	def __sub__(self, other):
		return (self.sprite.x - other.sprite.x, self.sprite.y - other.sprite.y)

	def draw(self):
		self.sprite.draw()

	def update(self):
		self.sprite.x = self.posx
		self.sprite.y = self.posy