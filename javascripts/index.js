const myCanvas = document.getElementById("canvas")
const c = myCanvas.getContext("2d")

const upperBound = 490
const lowerBound = 10

const redArc = new Circle(10, 10, 10, "red")
const centerArc = new Circle(230,230,20, "black")

function Circle(x, y, radius, color) {
	this.x = x
	this.y = y
	this.radius = radius
	this.color = color

	this.draw = function() {
		c.beginPath()
		c.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
		c.fillStyle = this.color
		c.fill()
	}

	this.moveRight = function() {
		if(this.x == 490)
			this.x = 10
		this.x += 1
		this.draw()
	}

	this.moveLeft = function() {
		if(this.x != 10)
			this.x -= 1
		this.draw()
	}
	
	this.moveUp = function() {
		if(this.y != 10)
			this.y -= 1
		this.draw()
	}
	
	this.moveDown = function() {
		if(this.y == 490)
			this.y = 10
		this.y += 1
		this.draw()
	}
}

