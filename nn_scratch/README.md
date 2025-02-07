# Micrograd Implementation by Andrej Karpathy

Learn the core concepts of neural networks and backpropagation with the implementation of *Micrograd* by [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy). The following resources will guide you:

- [The Spelled-out Intro to Neural Networks and Backpropagation: Building Micrograd (YouTube)](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)
- [Micrograd on GitHub](https://github.com/karpathy/micrograd/tree/master)

# Short Review of the Video

In this video, Andrej Karpathy walks viewers through the basics of neural networks by building a simple backpropagation algorithm from scratch. He explains how to apply this algorithm to build a simple Multi-Layer Perceptron (MLP).

## Getting Intuition on Derivatives

The video begins by helping viewers build an intuitive understanding of derivatives. Karpathy uses simple examples to explain the concept:

- **What are derivatives?** They represent the slope of a function and show how much one value will change when another value is slightly adjusted.
  
### Example
If `a = 2.0`, `b = -3.0`, and `c = 10.0`, then:
```python
d = a * b + c  # d = 4.0​

# Now, if we increase `a` variable slightly by h = 0.0001:
d1 = a * b + c  # d1 = 4.0
d2 = (a + h) * b + c  # d2 = 4.0001

# The derivative is calculated as: (f(x+h)−f(x)) / h
slope = 0.9999999999976694
# This means that increasing `a` by 0.0001 results in an increase of 0.0001 in `d`.
```

## Micrograd implementation
### `Value` Class: Simple Automatic Differentiation

The `Value` class is used for tracking values, operations, and gradients in a computation graph, which is essential for automatic differentiation. This is useful for tasks like training neural networks.

#### Overview

Each instance of the `Value` class represents a value in the computation graph. It supports common mathematical operations and provides a way to calculate gradients using backpropagation.

##### Attributes:
- **`data`**: The actual value stored in the object (e.g., a number).
- **`grad`**: The gradient (derivative) of the value, initially set to `0.0`.
- **`_backward`**: A function placeholder for backpropagation, which is set to an empty function by default.
- **`_prev`**: A set of previous values that contributed to this value (needed for gradient calculation).
- **`_op`**: The operation used to create this value (e.g., `+`, `*`, etc.).
- **`label`**: A label for easier identification (inspecting using `drawdot` explained further) (optional, like "ReLU" or "exp").

##### Supported Operations

The `Value` class supports several mathematical operations, such as addition, multiplication, exponentiation, and more. These operations build a computation graph, where each operation creates a new `Value` object and links it to the previous values.

##### Methods:

- **`__add__(self, other)`**: Adds two values together and returns a new `Value` object.

- **`__mul__(self, other)`**: Multiplies two values and works similarly to addition, but for multiplication.

- **`__pow__(self, other)`**: Raises a value to the power of another (e.g., `x ** 2`). Only supports integer or float exponents.

- **`_backward()`**: Sets up the backward function for gradient calculation during backpropagation for all operations that need it.

- **Reverse Operations**:
  - **`__rmul__`, `__radd__`, `__rsub__`, `__truediv__`, `__rtruediv__`, `__neg__`, `__sub__`**: These handle cases where the left operand isn't a `Value` object (e.g., multiplying a number by a `Value`).

- **Activation Functions**:
  - **`tanh(self)`**: Applies the hyperbolic tangent function and returns a new `Value` object.
  - **`relu(self)`**: Applies the ReLU function, returning 0 for negative inputs and the input itself for positive ones.

- **`exp(self)`**: Applies the exponential function (`e^x`), returning a new `Value` object with the result.

- **`backward(self)`**: This method performs backpropagation by traversing the computation graph, calculating gradients for all nodes in the graph.

### How It Works

When performing an operation (e.g., adding or multiplying two values), a new `Value` object is created. The operation defines a backward function that updates the gradients of the operands when `backward()` is called.

#### Example

```python
x = Value(2.0)
y = Value(3.0)
z = x * y + 1.0  # z = x * y + 1
z.backward()  # Calculates gradients for x, y, and z
```


## Visualization of Computation Graph with `graphviz`

This project uses the `graphviz` package to visualize the computation graph of `Value` objects, showing how values are computed and how gradients are propagated during backpropagation.

### Purpose

The goal is to generate a graphical representation of operations and dependencies between `Value` objects. This visualization helps understand the flow of data and gradients through the network.

### How It Works

The code generates a `graphviz` Digraph based on the computation graph of `Value` objects. The graph shows:

- The `data` and `grad` of each `Value` object.
- The operations (such as addition, multiplication) that generated each `Value`.
- The relationships (dependencies) between different `Value` objects.

### Functions

#### `trace(root)`
- Recursively builds a set of nodes and edges in the computation graph starting from the given root node.
- The function collects all nodes and edges of the graph by traversing the previous nodes (`_prev`).

#### `draw_dot(root)`
- Uses `graphviz` to create a graphical representation of the computation graph.
- **Attributes**:
  - **`rankdir='LR'`**: Specifies that the graph should be drawn from left to right.
  - **Nodes**: Each `Value` object is represented as a rectangular node, displaying its label, `data`, and `grad`.
  - **Edges**: Connects the nodes to visualize the flow of data and operations.
  - **Operations**: If a `Value` is the result of an operation, it creates an operation node and connects it to the corresponding value.

### How to Use

1. **Create `Value` Objects**: 
   Define `Value` objects and perform operations like addition or multiplication on them.

2. **Call `draw_dot`**: 
   Use the `draw_dot` function to generate the computation graph starting from the root `Value`.

### Example

```python
# Assuming the Value class is already defined

# Create some values and perform operations
x = Value(2.0, label="x")
y = Value(3.0, label="y")
z = x * y + 1.0  # z = x * y + 1

# Draw the computation graph
draw_dot(z)
```

## Multi-Layer Perceptron (MLP) Neural Network

This repository contains a basic implementation of a **Multi-Layer Perceptron (MLP)** neural network with forward and backward passes, training using **gradient descent**. The network is designed to handle simple binary classification tasks. Below is an explanation of the code and its components.

### Components

#### 1. **Neuron Class**
The `Neuron` class represents a single neuron in the neural network.

- **Initialization (`__init__`)**: 
  - The neuron has a set of **weights** (`w`) and a **bias** (`b`), both initialized with random values between -1 and 1.
  - `nin` is the number of inputs to the neuron.

- **`__call__` method**:
  - Takes an input `x` (a list of values), computes the weighted sum of the inputs (`w * x + b`), and applies the **tanh** activation function to the result.
  - Returns the activated output.

- **`parameters` method**:
  - Returns a list of the neuron’s weights and bias, which are the **trainable parameters** that will be updated during training.

#### 2. **Layer Class**
The `Layer` class represents a single layer of neurons in the neural network.

- **Initialization (`__init__`)**:
  - Takes the number of **inputs** (`nin`) and the number of **neurons** (`nout`) in the layer.
  - Creates a list of `nout` neurons, each having `nin` inputs.

- **`__call__` method**:
  - Passes the input `x` through each neuron in the layer and returns their outputs. If there's only one neuron, it returns that output; otherwise, it returns all the neuron outputs as a list.

- **`parameters` method**:
  - Returns the parameters (weights and bias) of all neurons in the layer.

#### 3. **MLP (Multi-Layer Perceptron) Class**
The `MLP` class represents the entire neural network, consisting of multiple layers.

- **Initialization (`__init__`)**:
  - Takes the number of **input features** (`nin`) and a list of the number of neurons in each layer (`nouts`).
  - Creates a sequence of layers based on the sizes provided.

- **`__call__` method**:
  - Passes the input `x` through all the layers in the network, one by one.

- **`parameters` method**:
  - Returns all the parameters (weights and biases) of the network by collecting them from each layer.

### 4. **Training the Network**
- **Inputs (`xs`)**: A set of example inputs for the network to learn from. Each input is a list of 3 values (features).
- **Targets (`ys`)**: The desired outputs (ground truth) for each corresponding input. These are the values the network is trying to predict.

#### Training Loop
The network is trained for 20 iterations, with the following steps:

1. **Forward pass**:
   - For each input, the network computes a predicted output (`y_pred`) using the `__call__` method.

2. **Loss Calculation**:
   - The loss is calculated as the sum of squared differences between the predicted outputs (`y_pred`) and the target outputs (`ys`).

3. **Backward pass**:
   - Gradients are computed for each parameter using `loss.backward()`. This step uses **backpropagation** to compute how much each parameter contributed to the loss.

4. **Parameter Update**:
   - The parameters (weights and biases) are updated by subtracting a fraction of their gradient (scaled by a learning rate of `0.05`).

5. The **loss** is printed after each iteration to track the model's performance.

### Final Prediction
After training, the model predicts the outputs for all the input examples (`xs`). The predictions (`y_pred`) will be printed out at the end.

### Key Concepts

- **Forward Pass**: The process of feeding inputs through the network to obtain the output.
- **Loss Function**: The loss function measures how far off the predicted outputs are from the true outputs. In this case, it is the **mean squared error (MSE)**.
- **Backpropagation**: The process of computing gradients of the loss with respect to the parameters (weights and biases), which are used to adjust the parameters.
- **Gradient Descent**: The method of updating parameters by moving them in the direction of the negative gradient of the loss, scaled by a learning rate.

### Example Flow

Given the input data `xs` and the desired target values `ys`, after training, the neural network will make predictions that should be closer to the target values. For each input in `xs`, the predictions (`y_pred`) will be printed, showing how the network learned to approximate the targets.

### Example Output

The training loop will print the **iteration number** and the **loss** at that iteration. After 20 iterations, the loss should decrease, indicating that the model is learning.

After training, the predictions for the inputs will be displayed in the form of `y_pred`.

### Conclusion

This simple MLP implementation provides the foundation for understanding how neural networks are built, trained, and evaluated. It can be extended to more complex tasks, deeper networks, and various types of activation functions and loss functions.
