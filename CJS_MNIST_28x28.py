import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, cache=True)
list(mnist)
mnist.data.shape
mnist.target.shape
plt.imshow(mnist.data[100].reshape(28,28),cmap='gray')

images = (0.99 * mnist.data + 0.01)/ 255

def sigmoid(x):
    return 1/(1+np.exp(-x))

num = mnist.target.shape[0]
print(num)

# one-hot encode the y_true values
y_true = np.zeros((num,10))
for i in range(num):
    y_true[i,int(mnist.target[i])] = 1

X_train,X_test,y_train,y_test = train_test_split(mnist.data,
                 y_true,
                 test_size = 0.8, 
                 random_state = 42)
size = X_train.shape[0]
print(size)

input_layer = 784
hidden_layer = 64
output_layer = 10

w1 = np.random.normal(0,
                      1/np.sqrt(input_layer),
                      size = (hidden_layer,input_layer))
w2 = np.random.normal(0,
                      1/np.sqrt(hidden_layer),
                      size = (output_layer,hidden_layer))
b1 = np.ones((hidden_layer,1))
b2 = np.ones((output_layer,1))


lr = .001

epochs = [x for x in range(50)]
Error = 0
for epoch in epochs:
    error = 0
    cnt = 0
    for i in range(size):
        # get the true value
        t = y_train[i].reshape(output_layer,1)
        # set nodes
        x0 = X_train[i].reshape(input_layer,1)
        x1 = sigmoid(w1.dot(x0)+b1)
        x2 = sigmoid(w2.dot(x1)+b2)
        # calculate errors
        error += np.sum((x2 - t)*(x2 - t))
        derr = (x2 - t)
        
        #calculate deltas
        delta_layer2 = (derr * (x2 * (1-x2)))        
        delta_layer1 = (((w2.T).dot(delta_layer2)) * (x1 * (1-x1)))
        
        #clean up notation
        w2 = w2 - lr * delta_layer2.dot(x1.T)
        b2 = b2 - lr * delta_layer2
        w1 = w1 - lr * delta_layer1.dot(x0.T)
        b1 = b1 - lr * delta_layer1
        #calculate how many are correct, for accuracy
        cnt += np.argmax(x2) == np.argmax(t)
    if epoch%1 == 0:
        print(f'Epoch {epoch}; Error is {error:.5f}; Accuracy is {100*cnt/size:.2f}%')
print("---------------------------------------------")
print(f'Final training accuracy is {100*cnt/size:.2f}%')

cnt = 0
print(f'Number of test samples is: {num-size}')
for i in range(num-size):
    t = y_test[i].reshape(output_layer,1)
    # set nodes
    x0 = X_test[i].reshape(input_layer,1)
    x1 = sigmoid(w1.dot(x0)+b1)
    x2 = sigmoid(w2.dot(x1)+b2)
    cnt += np.argmax(x2) == np.argmax(t)
print(f'Testing accuracy is {100 * cnt/(num-size):.2f}%')

print('Please input a digit:')
i = int(input())
blind = X_test[i].reshape(28,28)
plt.imshow(blind)
x0 = X_test[i].reshape(input_layer,1)
x1 = sigmoid(w1.dot(x0)+b1)
x2 = sigmoid(w2.dot(x1)+b2)
print(f'The queried digit is {np.argmax(x2)}.  The correct digit is {np.argmax(y_test[i])}')

plt.imshow(w2,cmap='gray')

plt.imshow(w1,aspect=10,cmap='gray')
