# ML_toy_examples
## my machine learning toy examples

### Here's my toy NN that attempts to model trigonometric functions with a single hidden layer neural network.  The activation function used for this example was the sigmoid function.  I got the idea for doing this from reading about the Universal Approximation Theorem.

### This is the sigmoid function and its derivative:

![sigmoid](https://github.com/cjsutton77/ML_toy_examples/blob/master/Unknown-2.png)

### Let's try and see if we can reproduce the Cosine squared function.  I ran my range from (0 - pi/2).

![What we want to model](https://github.com/cjsutton77/ML_toy_examples/blob/master/Unknown.png)

### We will feed the network using batch gradient descent.  So we'll generate a set of 500 numbers distributed from zero to half of pi.  

```python
x_train = np.random.uniform(0,np.pi/2,500)
```

### Then we'll feed this through the system and update weights and biases.

### Be sure to include backpropagation for the weights and biases.

```python
        d_err_d_ypred = -2 * (ytrue-ypred)

        d_ypred_d_w2 = h * deriv_sigmoid(np.matmul(h,w2)+b2)

        d_ypred_d_b2 = deriv_sigmoid(np.matmul(h,w2)+b2)

        d_ypred_d_h = w2 * deriv_sigmoid(np.matmul(x,w1)+b1).T

        d_h_d_w1 = x * deriv_sigmoid(np.matmul(x,w1)+b1)

        d_h_d_b1 = deriv_sigmoid(np.matmul(x,w1)+b1)

        w2 = w2 - learning_rate * np.matmul(d_err_d_ypred, d_ypred_d_w2).T
        w1 = w1 - learning_rate * np.matmul(d_err_d_ypred, d_ypred_d_h.T) * d_h_d_w1
        b2 = b2 - learning_rate * np.matmul(d_err_d_ypred, d_ypred_d_b2)
        b1 = b1 - learning_rate * np.matmul(d_err_d_ypred, d_ypred_d_h.T) * d_h_d_b1
```

### I wanted to this to be as clear as possible so that's why I've expressed the backpropagation in terms of the partial derivatives.

### This small toy example runs in about one minute.  Here's what you get - not perfect but not bad!

[!meh](https://github.com/cjsutton77/ML_toy_examples/blob/master/Unknown-4.png)

