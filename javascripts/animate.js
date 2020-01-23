
function animate() {
	//refresh the grid
	c.clearRect(0, 0, myCanvas.clientWidth, myCanvas.clientHeight)

	centerArc.draw()


	
	//move
	redArc.moveRight()
	redArc.moveDown()





	//makes recursive
	requestAnimationFrame(animate)
}

animate()
