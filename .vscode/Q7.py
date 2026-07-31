import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,MaxPooling2D,Conv2D,Dropout
import matplotlib.pyplot as plt


(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()

X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0

Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)


print(f"Training Data Shape : {X_train.shape},{Y_train.shape}")
print(f"Testing Data Shape : {X_test.shape},{Y_test.shape}")

model=Sequential([
      Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),
      MaxPooling2D(2,2),
      Conv2D(64,(3,3),activation="relu"),
      MaxPooling2D(2,2),
      Flatten(),
      Dense(128, activation="relu"),
      Dropout(0.5),
      Dense(10,activation="softmax"),
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

history=model.fit(
    X_train,Y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss,accuracy=model.evaluate(X_test,Y_test,verbose=0)
print(f"Baseline Model Test Accuracy : {accuracy:.4f}")


# IMPROVED MODEL

improved_model=Sequential([
      Conv2D(64,(5,5),activation="relu",input_shape=(32,32,3)),
      MaxPooling2D(2,2),
      Conv2D(128,(5,5),activation="relu"),
      MaxPooling2D(2,2),
      Flatten(),
#       Dense(256, activation="relu"),
#       Dropout(0.5),
#       Dense(10,activation="softmax"),
# ])

# optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)

# improved_model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

# improved_model.summary()

# improved_history=improved_model.fit(
#     X_train,Y_train,
#     epochs=20,
#     batch_size=64,
#     validation_split=0.2,
#     verbose=1
# )

# improved_loss,improved_accuracy=improved_model.evaluate(X_test,Y_test,verbose=0)
# print(f"Improved Model Test Accuracy : {improved_accuracy:.4f}")

# plt.plot(improved_history.history['loss'],label="Training Loss")
# plt.plot(improved_history.history['val_loss'],label="Validation Loss")
# plt.title("Loss Over Epochs")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()