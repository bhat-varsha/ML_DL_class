import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#tensor flow utilizes the graph and build the model and then run the model on the data
#can wokr in 3d data 


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0  #-1 indicate all samples , 1 indicate the number of channels
#channels
#astype :


y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  #convolutional layer 1
    MaxPooling2D(pool_size=(2, 2)),   #pooling layer 1
    Conv2D(64, (3, 3), activation='relu'),   #convolutional layer 2
    MaxPooling2D(pool_size=(2, 2)),    #pooling layer 2
    Flatten(),        #flattening the 2D arrays  dminesuin reduction 
    Dense(128, activation='relu'),     #fully connected layer
    Dense(10, activation='softmax')         #output layer for 10 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")