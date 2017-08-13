import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z))


class NeuralNetwork:

    def __init__(self, layers_size):

        self.layers_size = layers_size
        self.layers_num = len(self.layers_size)

        # Initialize weights and bias for each layer with random numbers
        self.weights = [np.random.random(size=(self.layers_size[i + 1], self.layers_size[i]))
                        for i in range(self.layers_num - 1)]

        self.biases = [np.random.random(size) for size in self.layers_size[1:]]

    def predict(self, x):

        a = self.__prepare_input(x)

        for w, b in zip(self.weights, self.biases):

            a = sigmoid(np.sum(w * a.T, axis=1) + b)

        return a

    def __prepare_input(self, x):

        if type(x) != np.ndarray:

            try:
                x = np.asarray(x)

            except ValueError as e:
                raise ValueError("Input data must be convertible to np.ndarray: {}".format(e))

        if x.shape != (self.layers_size[0],):

            raise ValueError("Size of input vector ({}) doesn't match network input layer ({})".format(
                             x.shape, (self.layers_size[0],)))

        return x


if __name__ == '__main__':

    nn = NeuralNetwork(layers_size=[3, 2, 4, 1])

    print(nn.predict(np.asarray([2, 0, 7])))
