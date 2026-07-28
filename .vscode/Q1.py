from tensorflow.keras.datasets import mnist,cifar10
import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# LOAD MNIST

(X_train_mnist,Y_train_mnist),(X_test_mnist,Y_test_mnist)=mnist.load_data()
print(f"MNIST-Training Shape : {X_train_mnist.shape}")
print(f"MNIST-Testing Shape : {X_test_mnist.shape}")

# LOAD CIFAR

(X_train_cifar,Y_train_cifar),(X_test_cifar,Y_test_cifar)=cifar10.load_data()
print(f"\nCIFAR-Training Shape : {X_train_cifar.shape}")
print(f"CIFAR-Testing Shape : {X_test_cifar.shape}")


layer=tf.keras.layers.Dense(units=10,activation='relu')
print(f"Tensorflow Layer : {layer}")


layer=nn.Linear(in_features=10,out_features=5)
print(f"Torch Layer : {layer}")


plt.imshow(X_train_mnist[0],cmap='gray')
plt.title(f"MNIST-Label : {Y_train_mnist[0]}")
plt.show()


plt.imshow(X_train_cifar[0],cmap='gray')
plt.title(f"Cifar-10 Label : {Y_train_cifar[0]}")
plt.show()