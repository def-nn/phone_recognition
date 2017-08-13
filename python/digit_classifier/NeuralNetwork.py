import numpy as np
import random


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z))


class NeuralNetwork:

    def __init__(self, layers_size):

        self.layers_size = layers_size
        self.layers_num = len(self.layers_size)

        np.random.seed(1)

        # Initialize weights and bias for each layer with random numbers
        self.weights = [np.random.random(size=(self.layers_size[i + 1], self.layers_size[i]))
                        for i in range(self.layers_num - 1)]

        self.biases = [np.random.random(size) for size in self.layers_size[1:]]

    def backpropagation(self, x, y):

        pass

    def update_mini_batch(self, mini_batch, alpha):

        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y)

            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.weights = [w - (alpha / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (alpha / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):

        pass

    def SGD(self, training_data, epochs, mini_batch_size, alpha=0.01, test_data=None):

        if test_data: n_test = len(test_data)

        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def predict(self, x):

        a = self.__prepare_input(x)

        for w, b in zip(self.weights, self.biases):

            a = sigmoid(np.dot(w, a) + b)

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
