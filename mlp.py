from tensor import Tensor # symlink from tensor-math
from math import exp # only for exp (e)

class MLP():

	def __init__(self, inputs: int, depth: int, layers: int, outputs: int, learning_rate: float):

		self.learning_rate = learning_rate

		# init dimensions
		self.inputs = inputs
		self.layers = layers
		self.neurons = depth 
		self.outputs = outputs

		# add i -> h weights, biases
		self.weights = [Tensor((self.neurons, self.inputs))]
		self.biases = [Tensor((self.neurons, 1))]

		if self.layers > 1: # add h -> h weights, biases
			for layer in range(self.layers -1):
				self.weights.append(Tensor((self.neurons, self.neurons)))
				self.biases.append(Tensor((self.neurons, 1)))

		# add h -> o weights, biases
		self.weights.append(Tensor((self.outputs, self.neurons)))
		self.biases.append(Tensor((self.outputs, 1)))

		# randomize all weights
		for tensor in self.weights:
			tensor.randomize()

		for bias in self.biases:
			bias.randomize()

	# wrapper activation / derivative activation functions
	def activation(self, x: float) -> float:
		return self.sigmoid(x)

	def d_activation(self, x: float) -> float:
		return self.de_sigmoid(x)

	# TODO: add tanh, relu
	def sigmoid(self, x: float) -> float:
		if x >= 0:
			return 1 / (1 + exp(-x))
		else:
			# save overflow if neg
			ex = exp(x)
			return ex / (1 + ex)
	
	def de_sigmoid(self, x: float) -> float:
		return x * (1 - x) # expects already sigmoided x

	def feed_forward(self, inputs: list):

		# input_matrix = Matrix(len(inputs), 1) # input conversion to matrix
		# input_matrix = input_matrix.init_from_array(inputs)

		input_tensor = Tensor.from_array(inputs)

		activations = [input_tensor]

		# iterate through every weight matrix and calculate the activations
		for idx, matrix in enumerate(self.weights):

			result = matrix @ input_tensor
			result = result + self.biases[idx]
			result = result.map(self.activation)
			
			activations.append(result)
			input_tensor = result # set for next

		return activations 

	def predict(self, inputs: list) -> list:
		return self.feed_forward(inputs)[-1].to_array()

	# verbose, long winded, modular impl for understanding
	def train(self, inputs: list, target_list: list):

		targets = Tensor((len(target_list), 1))
		targets = targets.from_array(target_list)

		# perform forward pass, keep input, neuron activations, and output (0,n,-1)
		activations = self.feed_forward(inputs)

		# compute output layer new weights and biases
		output = activations[-1] # activations of the ff output
		d_activated_output = output.map(self.d_activation) # save a version of the derivative of theactivated ouptut

		errors = targets.ew_sub(output) # calc result vs target result differences as the errors
		delta = d_activated_output.ew_mul(errors) # calc delta (error scaled by activation derivative)
		delta = delta.scalar_mul(self.learning_rate) # multiply by lr to get how big a step the model should take of the delta towards the target

		prev_activations = activations[-2].transpose() # transpose prev act for mat mul of weights
		weight_delta = delta.mat_mul(prev_activations) # mat mul the delta by the previous activations to get the final weight delta for this step

		# may switch this to after the loop
		self.weights[-1] = self.weights[-1].ew_add(weight_delta) # update the weights with the delta relative to prev layer
		self.biases[-1] = self.biases[-1].ew_add(delta)  # update the biases with the delta * 1 (as if its prev activations were always 1)

		# iterate through the hidden layers backwards and propagate the error
		# we already did the output layer + its prev activation, so we start there
		for l in reversed(range(len(self.weights) - 1)):

			# propagate error back through weights
			errors = self.weights[l+1].transpose()
			errors = errors.mat_mul(delta)

			# compute delta for this layer (activations[l+1] is current activation:
			# because we are only looping hidden layers but all have activations)
			delta = activations[l+1].map(self.d_activation)
			delta = delta.ew_mul(errors)
			delta = delta.scalar_mul(self.learning_rate)

			# weight delta uses previous layers activations (activations[l])
			prev_activations = activations[l].transpose()
			weight_delta = delta.mat_mul(prev_activations)

			self.weights[l] = self.weights[l].ew_add(weight_delta)
			self.biases[l] = self.biases[l].ew_add(delta)










