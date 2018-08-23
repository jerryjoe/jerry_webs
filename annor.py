# jerry_webs

# Backpropagation algorithm to implement an OR gate .

import string
import math
import random


class Ann:
    def __init__(self, neural_pattern):
        # we are going to use 2 input nodes, 2 hidden nodes and 1 output node.
        # Therefore, number of nodes in input(input_node)=2, hidden(hidden_node)=2, output(output_node)=1.
        self.input_node = 2
        self.hidden_node = 2
        self.output_node = 1

        # Initializing node weights. We make a two dimensional array that maps node from one layer to the next.
        # j-th node of one layer to k-th node of the next.
        #
        self.hidden_input_weight = []
        for j in range(self.input_node):
            self.hidden_input_weight.append([0.0] * self.hidden_node)

        self.hidden_output_weight = []
        for k in range(self.hidden_node):
            self.hidden_output_weight.append([0.0] * self.output_node)

        # Initializing, the activation matrices.

        self.a_input, self.a_hidden, self.a_output = [], [], []
        self.a_input = [1.0] * self.input_node
        self.a_hidden = [1.0] * self.hidden_node
        self.a_output = [1.0] * self.output_node

        #
        # To ensure node weights are randomly assigned, with some bounds on values, we pass it through randomizeMatrix()
        #
        randomWeights(self.hidden_input_weight, -0.1, 0.1) # for random input initial weights
        randomWeights(self.hidden_output_weight, -0.5, 0.2) # for random output initial weights

        # Another array to incorporate the previous change.
        self.cih = []
        self.cho = []
        for i in range(self.input_node):
            self.cih.append([0.0] * self.hidden_node)
        for j in range(self.hidden_node):
            self.cho.append([0.0] * self.output_node)

    # backpropagate() function takes as input, the patterns entered, the target values and the obtained values.
    # for which it uses to adjusts the weights so as to balance out the error.
    # we also have M, N for momentum and learning factors respectively.
    def back_propagate(self, inputs, expected, output, N=0.5, M=0.1):
        # We introduce a new matrix called the deltas (error) for the two layers output and hidden layer respectively.
        output_deltas = [0.0] * self.output_node
        for k in range(self.output_node):
            # Error is  (Target - Output)
            error = expected[k] - output[k]
            output_deltas[k] = error * dsigmoid(self.a_output[k])

        # Changing the weights of hidden to output layer accordingly.
        for j in range(self.hidden_node):
            for k in range(self.output_node):
                delta_weight = self.a_hidden[j] * output_deltas[k]
                self.hidden_output_weight[j][k] += M * self.cho[j][k] + N * delta_weight
                self.cho[j][k] = delta_weight

        # Now for the hidden layer.
        hidden_deltas = [0.0] * self.hidden_node
        for j in range(self.hidden_node):
            # Error as given by formule is equal to the sum of (Weight from each node in hidden layer times output delta of output node)
            # Hence delta for hidden layer = sum (self.who[j][k]*output_deltas[k])
            error = 0.0
            for k in range(self.output_node):
                error += self.hidden_output_weight[j][k] * output_deltas[k]
            # now, change in node weight is given by dsigmoid() of activation of each hidden node times the error.
            hidden_deltas[j] = error * dsigmoid(self.a_hidden[j])

        for i in range(self.input_node):
            for j in range(self.hidden_node):
                delta_weight = hidden_deltas[j] * self.a_input[i]
                self.hidden_input_weight[i][j] += M * self.cih[i][j] + N * delta_weight
                self.cih[i][j] = delta_weight

    # Function for testing after training and backpropagation is complete.
    def test_fn(self, p_data):
        for p in p_data:
            inputs = p[0]
            print('For input:', p[0], ' Output -->', self.runNetwork(inputs), '\tExpected Target: ', p[1])

    # So, runNetwork was needed because, for every iteration over a p_data [] array, we need to feed the values.
    def runNetwork(self, feed):
        if (len(feed) != self.input_node):
            print('The number of input values are erroneous!')

        # First activate the input nodes.
        for i in range(self.input_node):
            self.a_input[i] = feed[i]

        #
        # Calculate the activations of each successive layer's nodes.
        #
        for j in range(self.hidden_node):
            sum = 0.0
            for i in range(self.input_node):
                sum += self.a_input[i] * self.hidden_input_weight[i][j]
            # self.a_hidden[j] will be the sigmoid of sum. # sigmoid(sum)
            self.a_hidden[j] = sigmoid(sum)

        for k in range(self.output_node):
            sum = 0.0
            for j in range(self.hidden_node):
                sum += self.a_hidden[j] * self.hidden_input_weight[j][k]
            # self.ah[k] will be the sigmoid of sum. # sigmoid(sum)
            self.a_output[k] = sigmoid(sum)
            for i in self.a_output: #After 10000 epochs, the model returns a value of less than 0.8 for combination [0,0], hence otherwise.Part of the activation function.
                if i <= 0.8:
                    new_i = [0]
                else:
                    new_i = [1]

        return new_i

    def train_Network(self, pat):
        for i in range(10000):
            # Run the network for every set of input values, get the output values and Backpropagate them.
            for p in pat:
                # Run the network for every tuple in p.
                inputs = p[0]
                out = self.runNetwork(inputs)
                expected = p[1]
                self.back_propagate(inputs, expected, out)
        self.test_fn(pat)


# End of class.

# matrix for assigning random weights to the inputs.
def randomWeights(matrix, x, y):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # For each of the weight matrix elements, assign a random weight uniformly between the two bounds.
            matrix[i][j] = random.uniform(x, y)


# Now for our function definition. Sigmoid.
def sigmoid(x):
    return 1 / (1 + math.exp(-x))  # implementing the sigmoid activation function.(1/1+e^-x)


# Sigmoid function derivative.
def dsigmoid(y):
    return y * (1 - y)  # Derivative of the sigmoid function


def main():
    # take the input pattern as a map. Suppose we are working for AND gate.
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [1]]
    ]
    newNeural = Ann(pat)
    newNeural.train_Network(pat)


if __name__ == "__main__":
    main()


"""THE OUTPUT IS AS FOLLOWS:
C:\Users\n\PycharmProjects\ANN\venv\Scripts\python.exe C:/Users/n/PycharmProjects/ANN/venv/Scripts/ANN1.py
For input: [0, 0]  Output --> [0] 	Expected Target:  [0]
For input: [0, 1]  Output --> [1] 	Expected Target:  [1]
For input: [1, 0]  Output --> [1] 	Expected Target:  [1]
For input: [1, 1]  Output --> [1] 	Expected Target:  [1]

Process finished with exit code 0"""
