import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, output_size, hidden_layers_size):

        self.layers_size = [input_size] + hidden_layers_size + [output_size]
        self.layers_num = len(self.layers_size)

        self.layers = [np.empty(shape=layer_size) for layer_size in self.layers_size]

        # Initialize weights and bias for each layer with random numbers
        self.weights = [np.random.random(size=(self.layers_size[i+1], self.layers_size[i]))
                        for i in range(self.layers_num - 1)]

        self.bias = [np.random.rand() for _ in range(self.layers_num - 1)]

    def predict(self, x):

        pass


if __name__ == '__main__':

    nn = NeuralNetwork(input_size=3,
                       output_size=1,
                       hidden_layers_size=[2, 4])
