class Config:
	def __init__(self):
		self.COLOR_POS = [255, 255, 255]
		self.COLOR_NEG = [255, 64, 0]

		self.FAST_STEP = 1000000
		self.SLOW_STEP = 100000
		self.FINE_STEP = 10000
		self.SUPERFINE_STEP = 1000

		self.cols = 7
		self.rows = 5
		self.GRID_SIZE = (self.cols, self.rows)