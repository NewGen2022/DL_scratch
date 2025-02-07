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

