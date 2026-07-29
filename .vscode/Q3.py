import numpy as np
import matplotlib.pyplot as plt

def mse_loss(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

def binary_count_entropy_loss(y_true,y_pred):
    y_pred=np.clip(y_pred,1e-15,1 - 1e-15)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

y_true=np.array([1,0,1,1])
y_pred=np.array([0.9,0.2,0.8,0.7])

mse=mse_loss(y_true,y_pred)
bce=binary_count_entropy_loss(y_true,y_pred)

print(f"MSE LOSS : {mse:.4f}")
print(f"BCE LOSS : {bce:.4f}")


def mse_gradient(y_true,y_pred):
    return 2 * (y_pred - y_true) /len(y_true)

def binary_count_entropy_gradient(y_true,y_pred):
    y_pred=np.clip(y_pred,1e-15,1 - 1e-15)
    return (y_pred - y_true)/(y_pred * (1 - y_pred))

grad_mse=mse_gradient(y_true,y_pred)
grad_bce=binary_count_entropy_gradient(y_true,y_pred)

print(f"MSE GRADIENTS : {grad_mse}")
print(f"BCE GRADIENTS : {grad_bce}")


predictions=np.linspace(0,1,100)
true_label=1

mse_losses=[(true_label - p)**2 for p in predictions]
bce_losses=[-true_label * np.log(max(p,1e-15)) - (1 - true_label )* np.log(max(p-1,1e-15)) for p in predictions]

plt.figure(figsize=(10,8))
plt.plot(predictions,mse_losses,label="MSE Loss")
plt.plot(predictions,bce_losses,label="BCE Loss")
plt.title("Loss Functions Comparison")
plt.xlabel("Prediction")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()