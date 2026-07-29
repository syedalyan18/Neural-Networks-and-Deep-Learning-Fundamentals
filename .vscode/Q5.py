import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Flatten,MaxPooling2D,Conv2D,Dropout
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

X_train=X_train.reshape(-1,28,28,1).astype('float32')/255.0
X_test=X_test.reshape(-1,28,28,1).astype('float32')/255.0

Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)


print(f"Training Data Shape : {X_train.shape}")
print(f"Testing Data Shape : {X_test.shape}")

model=Sequential([
      Conv2D(32,(3,3),activation="reLu",input_shape=(28,28,1)),
      MaxPooling2D(2,2),
      Flatten(),
      Dense(128,activation="reLu"),
      Dropout(0.5),
      Dense(10,activation="softmax"),
])

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=['accuract']
)

history=model.fit(
    X_train,Y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

test_lost,test_accuracy=model.evaluate(X_test,Y_test)
print(f"Test Accuracy : {test_accuracy:.4f}")

loaded_model=load_model('mnist_classifier.h5')

loss,accuracy=load_model.evaluate(X_test,Y_test)
print(f"Loaded Model Accuracy : {accuracy:.4f}")