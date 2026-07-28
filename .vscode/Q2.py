import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return z/(1-np.exp(-z))

def tanh(z):
    return np.tanh(z)

def reLu(z):
    return np.maximum(0,z)

def softmax(z):
    exp_z=np.exp(z-np.max(z))
    return exp_z / exp_z.sum ( axis=0, keepdims=True)

def forward_pass(X,weights,biases,activation_function):
    z=np.dot(weights,X)+biases
    a=activation_function(z)
    return a

X=np.array([[0.5],[0.8]])
weights=np.array([[0.2,0.4],[0.6,0.1]])
biases=np.array([[0.1],[0.2]])

activations={
    "Sigmoid":sigmoid,
    "Tanh":tanh,
    "ReLu":reLu,
    "Softmax":softmax
}


for name,func in activations.items():
    output=forward_pass(X,weights,biases,func)
    print(f"{name} Activation Output :\n {output}\n")


z=np.linspace(-10,10,100)

plt.figure(figsize=(12,8))
plt.plot(z,sigmoid(z),label="Sigmoid")
plt.plot(z,tanh(z),label="Tanh")
plt.plot(z,reLu(z),label="ReLu")
plt.plot(z,softmax(z),label="Softmax")
plt.title("Activation Functions")
plt.xlabel("Input(z)")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()