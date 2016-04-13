import random

#Using sigmoid as activation function
class Neu_Net( object ):

	#Initialize function
	def __init__(self):

		#Number of nodes for each layer
		self.input = 900
		self.hidden = 4
		self.output = 2

		#Initializing weights with random numbers. Since we have 3 layes, we have 2 matrices of weights.
		#weight_1 should be 900x4 
		#weight_2 should be 4x2
		self.weight_1 = np.random.randn( self.input , self.hidden )
		self.weight_2 = np.random.randn( self.hidden , self.output )

	#Foward propagation. X is matrix of inputs, which will be 50x900 because we have 50 30x30 pics
	def foward( self, X ):

		#dot method is matrix multiplication
		self.z2 = np.dot( X , self.weight_1 )
		self.a2 = self.sigmoid( self.z2 )
		self.z3 = np.dot( self.a2, self.weight_2 )
		hypo = self.sigmoid( self.z3 )
		return hypo

	def sigmoid( self, k ):

		#apply sigmoid to matrix
		return 1/( 1 + np.exp( -z ) )
