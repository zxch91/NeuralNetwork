Artificial Neural Network for Regression

This project implements a simple feedforward artificial neural network with one hidden layer. The network is designed for regression tasks and is trained using backpropagation, gradient descent, and optional momentum. The code also includes automatic learning rate adjustment using a bold driver technique. After training, the network evaluates performance on validation and test datasets and produces prediction plots.

Overview of the Model

The ANN class provides the following features:

Custom network architecture defined by:
• number of input nodes
• number of hidden nodes
• number of output nodes

Forward propagation using the sigmoid activation function.

Backward propagation to compute gradients for each layer.

Gradient descent weight updates with optional momentum.

Bold driver learning rate adjustment every 50 epochs.

Tracking and plotting of mean squared error across epochs.

Prediction plots for validation and test datasets.

Calculation of a simple accuracy metric based on total predicted and actual values.

Input Data

The program loads three Excel files:
• train_data.xlsx
• val_data.xlsx
• test_data.xlsx

Each file is expected to contain predictors in columns 2 through 6 and the predictand in column 7.

Running the Program

The program prompts the user for:
• number of predictors
• number of hidden nodes
• number of outputs

The ANN is created with these values.

Training runs for a fixed number of epochs with user-defined learning rate and momentum.

After training, the program:
• plots mean squared error over time
• plots predicted versus actual values for validation data
• evaluates the network on test data
• computes and prints an accuracy score

Key Methods

forward_pass
Computes layer activations from input to output.

backwards_pass
Computes errors and gradients for weight updates.

gradient_descent
Updates weights using gradients and momentum.

networkTrain
Runs full training across all epochs and produces evaluation plots.

sigmoid and derivative_sigmoid
Activation function and its derivative.

Outputs

During and after training, the program provides:
• printed MSE values per epoch
• plot of MSE vs epoch
• scatter plot of predicted vs actual validation values
• scatter plot of predicted vs actual test values
• accuracy value for test predictions

Dependencies

The program requires:
• numpy
• pandas
• matplotlib
• a Python environment capable of running standard scientific packages
