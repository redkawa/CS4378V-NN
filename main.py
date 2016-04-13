import random


class Neu_Net( object ):
	def __init__(self):

		#Number of nodes for each layer
		self.input = 900
		self.hidden = 4
		self.output = 2

		#Initializing weights. Since we have 3 layes, we have 2 matrices of weights. 
		self.weight_1 = np.random.randn( self.input , self.hidden )
		self.weight_2 = np.random.randn( self.hidden , self.output )


