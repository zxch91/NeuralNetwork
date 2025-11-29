# README  
## Artificial Neural Network for Regression

This project implements a simple feedforward artificial neural network with one hidden layer. The network is designed for regression tasks and is trained using backpropagation, gradient descent, and optional momentum. The code also includes automatic learning rate adjustment using a bold driver technique. After training, the network evaluates performance on validation and test datasets and produces prediction plots.

---

## Features

The `ANN` class includes:

- Customizable architecture  
  - number of input nodes  
  - number of hidden nodes  
  - number of output nodes  
- Forward propagation using the sigmoid activation function  
- Backward propagation to compute layer gradients  
- Gradient descent with optional momentum  
- Bold driver learning rate adjustment every 50 epochs  
- Tracking and plotting of mean squared error  
- Prediction plots for validation and test datasets  
- Simple accuracy metric comparing total predicted and actual values  

---

## Data Requirements

The program expects three Excel files:

- `train_data.xlsx`  
- `val_data.xlsx`  
- `test_data.xlsx`  

Each file must contain:

- Predictor columns in positions 2 through 6  
- Predictand column in position 7  

---

## Usage

1. Run the script.  
2. Provide the following values when prompted:  
   - Number of predictors  
   - Number of hidden nodes  
   - Number of outputs  
3. The script loads training, validation, and test data.  
4. Training begins using the specified number of epochs, learning rate, and momentum.  
5. After training:  
   - A plot of mean squared error across epochs is displayed  
   - A scatter plot of predicted vs actual values for validation data is shown  
   - The model is evaluated on test data  
   - Test predictions are plotted  
   - Accuracy is printed to the console  

---

## Key Methods

### `forward_pass(input)`
Computes activations from input to output.

### `backwards_pass(error)`
Computes deltas and gradients for backpropagation.

### `gradient_descent(learning_rate, momentum)`
Updates weights using gradient values and optional momentum.

### `networkTrain(...)`
Runs training across all epochs and generates evaluation plots.

### `sigmoid(x)` and `derivative_sigmoid(x)`
Activation function and its derivative.

---

## Output

During and after training, the program provides:

- Mean squared error printed per epoch  
- MSE vs epoch plot  
- Predicted vs actual scatter plot for validation data  
- Predicted vs actual scatter plot for test data  
- Accuracy percentage for test predictions  

---

## Dependencies

This project requires:

- Python 3.8 or later  
- numpy  
- pandas  
- matplotlib  

---

## Notes

This implementation is simplified and intended for educational purposes. It does not include features common in modern neural network frameworks such as regularization, adaptive optimizers, batch training, or alternative activation functions. It is best suited for small datasets.
