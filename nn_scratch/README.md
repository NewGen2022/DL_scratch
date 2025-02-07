# Micrograd Implementation by Andrej Karpathy

Learn the core concepts of neural networks and backpropagation with the implementation of *Micrograd* by [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy). The following resources will guide you:

- [The Spelled-out Intro to Neural Networks and Backpropagation: Building Micrograd (YouTube)](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2)
- [Micrograd on GitHub](https://github.com/karpathy/micrograd/tree/master)

## Short Review of the Video

In this video, Andrej Karpathy walks viewers through the basics of neural networks by building a simple backpropagation algorithm from scratch. He explains how to apply this algorithm to build a simple Multi-Layer Perceptron (MLP).

## Getting Intuition on Derivatives

The video begins by helping viewers build an intuitive understanding of derivatives. Karpathy uses simple examples to explain the concept:

- **What are derivatives?** They represent the slope of a function and show how much one value will change when another value is slightly adjusted.
  
### Example:
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
