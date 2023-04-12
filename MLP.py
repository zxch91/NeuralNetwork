import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class ANN:
    def __init__(self, inputs, hidden, output):
        # inputs - how many predictors we have
        # hidden - how many hidden nodes we have
        # output - how many outputs we have
        self.inputs = inputs
        self.hidden = hidden
        self.output = output
        self.prev_error = None
        self.learning_rate = None
        self.total_errors = []
        self.previous_weights = []
        nodesPerLayer = [self.inputs, self.hidden, self.output]
        placeholder = None
        # create placeholder values for the future activations and derivatives
        # creates it so if you printed the array you'd easily be able to see what links to what
        # if you was to draw it graphically
        self.activations = [np.full(num_neurons, placeholder) for num_neurons in nodesPerLayer]
        self.derivatives = [np.full((prev_layer_neurons, curr_layer_neurons), placeholder) for
                            prev_layer_neurons, curr_layer_neurons in zip(nodesPerLayer[:-1], nodesPerLayer[1:])]
        self.prev_weight_change = [np.full((prev_layer_neurons, curr_layer_neurons), placeholder) for
                            prev_layer_neurons, curr_layer_neurons in zip(nodesPerLayer[:-1], nodesPerLayer[1:])]

        np.random.seed() # set a random seed every time so we get different weights
        # when testing you could set a set seed to see how you improve as you change stuff in your model
        self.weights = []
        self.biases = []
        for layer in range(len(nodesPerLayer) - 1):
            # generate random weights and biases between 0 and 1
            weight = np.random.uniform(low=0, high=1, size=(nodesPerLayer[layer], nodesPerLayer[layer + 1]))
            bias = np.random.uniform(low=0, high=1, size=nodesPerLayer[layer + 1])
            self.weights.append(weight)
            self.biases.append(bias)
        # placeholder for future previous weights in our momentum improvements
        self.previous_weights = [np.full((prev_layer_neurons, curr_layer_neurons), placeholder) for
                            prev_layer_neurons, curr_layer_neurons in zip(nodesPerLayer[:-1], nodesPerLayer[1:])]


    def forward_pass(self, input):
        # set the input layer as the first activation
        self.activations[0] = input

        # perform a forward pass through the layers
        for layer_index in range(len(self.weights)):
            # get the weights for the current layer
            weights = self.weights[layer_index]

            # compute the weighted sum for the current layer
            weighted_sum = np.dot(self.activations[layer_index], weights) + self.biases[layer_index]

            # compute the activations for the current layer using the sigmoid function
            activation = self.sigmoid(weighted_sum)

            # store the activation function in the activations list
            self.activations[layer_index + 1] = activation

        # return the final output layer activation
        return self.activations[-1]

    def backwards_pass(self, error):
        # calculate the derivative of the output activation function
        output_derivative = self.derivative_sigmoid(self.activations[-1])

        # calculate the delta for the output layer
        delta = error * output_derivative

        # update the derivatives for the output layer, the outer function here helps a lot as it means we can
        # compute each layer(column in matrix/vector terms) very easily with our delta value
        self.derivatives[-1] = np.outer(self.activations[-2], delta)

        # propagate the delta backwards through the layers
        for layer_index in range(len(self.weights) - 2, -1, -1): # START FROM THE 2ND LAST LAYER AS WE'VE DONE THE FIRST
            # calculate the derivative of the activation function for the current layer
            derivative = self.derivative_sigmoid(self.activations[layer_index + 1])

            # calculate the delta for the current layer
            delta = np.dot(self.weights[layer_index + 1], delta) * derivative

            # update the derivatives for the current layer
            self.derivatives[layer_index] = np.outer(self.activations[layer_index], delta)

            # update the bias with the delta
            self.biases[layer_index] += delta
        return self.derivatives
    def gradient_descent(self, learning_rate, momentum):
        # update the weights for each layer based on the derivatives and learning rate
        for layer_index in range(len(self.weights)):

            if momentum > 0:
                # use of .any function checks if we have made the first pass so momentum can be applied
                if self.previous_weights[layer_index].any() is not None:
                    delta = learning_rate * self.derivatives[layer_index]
                    momentum_term = momentum * (delta - self.previous_weights[layer_index])
                else:
                    momentum_term = 0
                # update the weights with momentum
                delta = learning_rate * self.derivatives[layer_index] + momentum_term
                self.weights[layer_index] += delta
                # save the delta for momentum in the next iteration
                self.previous_weights[layer_index] = delta

    def networkTrain(self, predictors, predictands, validation_predictors,
              validation_predictands, epochs, learning_rate, momentum):
        for epoch in range(epochs):
            total_error = 0 # reset for each epoch
            for predictor, predictand in zip(predictors, predictands):
                # following steps 2 to 5 in the lectures
                res = self.forward_pass(predictor)
                error = predictand - res
                # squares the error so when it comes to plotting on the graph we can plot the mean squared error
                total_error += np.sum(error ** 2)
                # did try to combine the backwards pass and gradient descent into the same function when
                # developing but the algorithm performed significantly worse
                self.backwards_pass(error)
                self.gradient_descent(learning_rate, momentum)
            self.total_errors.append(total_error)

            print('the MSE per result at the end of this set of epochs is ', epoch, total_error / len(predictors))

            # implementation of bold driver

            if epoch % 50 == 0: # do it every 50 epochs so we dont snowball
                if learning_rate < 0.01 or learning_rate > 0.5:
                    if self.prev_error is not None:
                        if total_error > self.prev_error:
                            learning_rate *= 0.7
                        if total_error < self.prev_error:
                            learning_rate *= 1.05
            self.prev_error = total_error

        #Plot total error vs. epoch after training has completed
        epochs_to_plot = np.arange(0, epochs, 20)  # Plot every 20th value in total_errors
        plt.plot(epochs_to_plot, [self.total_errors[i] / len(predictors) for i in epochs_to_plot])
        plt.title("Total Error vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error at epoch")
        plt.show()

        # Get predicted values for model that has been trained
        predicted_values = [self.forward_pass(x) for x in validation_predictors]

        # Plot actual values on x-axis, predicted values on y-axis
        plt.scatter(validation_predictands, predicted_values)

        # Add a line of perfect prediction
        plt.plot([validation_predictands.min(), validation_predictands.max()], [validation_predictands.min(), validation_predictands.max()])

        # Add labels and title
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')

        # Show plot
        plt.show()

    def sigmoid(self, x):
        fx = 1.0 / (1 + np.exp(-x))
        return fx

    def derivative_sigmoid(self, x):
        return x * (1.0 - x)

inputs = int(input('Enter the number of predictors in your data set'))
hiddenLayer = int(input('Enter the number of hidden nodes within the hidden layer'))
outputs = int(input('Enter the number of predictands in your data set'))

mlp = ANN(inputs,hiddenLayer,outputs) # input, hidden nodes, output - order

# load in our 3 bundles of data into a dataframe each
train_data = pd.read_excel('train_data.xlsx')
validation_data = pd.read_excel("val_data.xlsx")
test_data = pd.read_excel("test_data.xlsx")

predictors = train_data.iloc[:, 1:6].values
predictand = train_data.iloc[:, 6].values
validationPredictors = validation_data.iloc[:, 1:6].values
validationsPredictands = validation_data.iloc[:, 6].values
testPredictors = test_data.iloc[:, 1:6].values
testPredictands = test_data.iloc[:, 6].values

mlp.networkTrain(predictors, predictand, validationPredictors, validationsPredictands, epochs=1000, learning_rate=0.7, momentum=0.9)

testDataPass = mlp.forward_pass(testPredictors)
plt.scatter(testPredictands, testDataPass)
plt.plot([testPredictands.min(), testPredictands.max()], [testPredictands.min(), testPredictands.max()])

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')

# Show plot
plt.show()

actualTotal = testPredictands.sum()
networkTotal = testDataPass.sum()

difference = abs(actualTotal-networkTotal)
bothtotal = actualTotal+networkTotal
accuracy = ((bothtotal-difference)/bothtotal) * 100
accuracy = round(accuracy, 2)
print('The accuracy of our neural network model is',accuracy,'%')