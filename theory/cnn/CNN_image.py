import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

model = Sequential([
    Conv2D(32,(3, 3),activation='relu',input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64,(3, 3),activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
# 4. DISPLAY MODEL SUMMARY
print(model.summary())
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# 6. TRAIN CNN
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Loss:", loss)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 10. TEST YOUR OWN HANDWRITTEN IMAGE
from PIL import Image, ImageOps
image_path = "6.jpeg"
img = Image.open(image_path)
img = img.convert("L")
img = img.resize((28, 28))
img_array = np.array(img)
img_array = 255 - img_array
img_array = img_array.astype("float32") / 255.0

img_array = img_array.reshape(1, 28, 28, 1)

# 11. CLASSIFY YOUR OWN HANDWRITTEN DIGIT
own_prediction = model.predict(img_array)
predicted_digit = np.argmax(own_prediction)
confidence = np.max(own_prediction) * 100

print("Predicted Digit :", predicted_digit)
print(f"Confidence      : {confidence:.2f}%")


# ==========================================================
# 12. DISPLAY OWN IMAGE AND PREDICTION
# ==========================================================
plt.imshow(
    img_array.reshape(28, 28),
    cmap='gray'
)
plt.title(
    f"Predicted Digit: {predicted_digit}\n"
    f"Confidence: {confidence:.2f}%"
)
plt.axis('off')
plt.show()