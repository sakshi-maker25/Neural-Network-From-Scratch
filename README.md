# Neural Network from Scratch for Two-Moons Classification

## Overview

This project implements a feedforward neural network from scratch using Python and NumPy to solve a nonlinear binary classification problem on the two-moons dataset.

The objective is to understand the mathematical foundations of neural networks by implementing forward propagation, backpropagation, and gradient descent without relying on high-level deep learning frameworks such as TensorFlow or PyTorch.

**Course:** Numerical Analysis for Machine Learning  
**Tools:** Python, NumPy, Matplotlib

## Project Objectives

- Implement a neural network using fundamental numerical operations.
- Understand nonlinear classification using the hyperbolic tangent (tanh) activation function.
- Implement forward propagation and backpropagation.
- Train the network using batch gradient descent.
- Visualize the learned decision boundary.

## Dataset

The project uses a synthetic two-moons dataset containing 500 samples with a noise level of 0.1.

The dataset consists of two interleaving crescent-shaped classes, making it suitable for evaluating nonlinear classification.

Class labels are represented as -1 and +1.

## Methodology

### 1. Neural Network Architecture

A fully connected neural network with one hidden layer is implemented using NumPy.

- Input layer: Two features
- Hidden layer: Tanh activation
- Output layer: Prediction of the target label

### 2. Forward Propagation

The network computes predictions through matrix multiplication, bias addition, and nonlinear activation functions.

### 3. Loss Function

Mean Squared Error (MSE) measures the difference between predicted and actual labels.

### 4. Backpropagation

Gradients are calculated using the chain rule to determine how each network parameter contributes to the prediction error.

### 5. Optimization

Batch gradient descent updates the network weights and biases to minimize the loss function.

### 6. Decision Boundary Visualization

The trained network is evaluated over a two-dimensional grid to visualize how it separates the two classes.

## Results

The project generates a decision map illustrating the nonlinear classification regions learned by the neural network.

The visualization demonstrates how a neural network with nonlinear activation functions can model decision boundaries that cannot be represented by a single linear separator.

## Key Learning Outcomes

- Understanding neural network computations at the matrix level.
- Applying differential calculus and the chain rule to backpropagation.
- Implementing numerical optimization through gradient descent.
- Understanding the importance of nonlinear activation functions.
- Visualizing neural network classification behavior.

## Technologies Used

- Python
- NumPy
- Matplotlib

## How to Run

1. Clone the repository.
2. Install the required Python libraries:

   ```bash
   pip install numpy matplotlib
   ```

3. Run the Python script:

   ```bash
   python halfmoons.py
   ```

4. View the generated decision boundary visualization.

## Future Improvements

- Experiment with different hidden-layer sizes and learning rates.
- Compare tanh with other activation functions.
- Investigate the effect of noise on classification performance.
- Evaluate accuracy on a separate test dataset.

## Author

Sakshi Matale  
M.Sc. Data Science — Technische Universität Ilmenau

**GitHub:** [sakshi-maker25](https://github.com/sakshi-maker25)
