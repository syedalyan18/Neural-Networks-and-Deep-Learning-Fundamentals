import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)

X=np.random.rand(100,1)
Y=4 + 3 * X + np.random.rand(100,1)

plt.scatter(X,Y,color="blue")
plt.title("Generated Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()


m=100
theta=np.random.rand(2,1)
learning_rate=0.1
iterations=1000

X_b=np.c_[np.ones((m,1)),X]

for iteration in range(iterations):
    gradients=2/m * X_b.T.dot(X_b.dot(theta) - Y)
    theta -= learning_rate * gradients

print("Optimized Parameters (theta) : ",theta)


X_tensor=tf.constant(X,dtype=tf.float32)
Y_tensor=tf.constant(Y,dtype=tf.float32)

class LinearModel(tf.Module):
    def __init__(self):
        self.weights=tf.Variable(tf.random.normal([1]))
        self.bias=tf.Variable(tf.random.normal([1]))

    def __call__(self,X):
        return self.weights * X + self.bias

def mse_loss(y_true,y_predict):
    return tf.reduce_mean(tf.square(y_true - y_predict))

model=LinearModel()
optimizer=tf.optimizers.SGD(learning_rate=0.1)

for epoch in range(100):
    with tf.GradientTape() as tape:
        y_pred=model(X_tensor)
        loss=mse_loss(Y_tensor,y_pred)

    gradients=tape.gradient(loss,[model.weights,model.bias])
    optimizer.apply_gradients(zip(gradients,[model.weights,model.bias]))

    if epoch % 10 ==0:
        print(f"Epoch {epoch}, Loss {loss.numpy():.4f}")



X_torch=torch.tensor(X,dtype=torch.float32)
Y_torch=torch.tensor(Y,dtype=torch.float32)


class LinearModelTorch(nn.Module):
    def __init__(self):
        super (LinearModelTorch,self).__init__()
        self.linear=nn.Linear(1,1)

    def forward(self,X):
        return self.linear(X)

model_torch=LinearModelTorch()
criterion=nn.MSELoss()
optimizer=optim.Adam(model_torch.parameters(),lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    outputs=model_torch(X_torch)
    loss=criterion(outputs,Y_torch)

    loss.backward()
    optimizer.step()

 
    if epoch % 10 ==0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")
