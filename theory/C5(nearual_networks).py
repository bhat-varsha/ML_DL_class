import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#generating synthetic dataset for binary classification
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#for higkty vairable data ,(which hav extreme outliers)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#sequential is the nerual netowkr algorithm which is used to create the model in a linear stack of layers
# we have put 16,8,1 those are nodes , we cant just put any one , we shld put it in multiple of eight
#neural netwros start point is to create a model and then we can add layers to it
model = Sequential([
    Dense(16, input_dim=2, activation='relu'),  # Hidden layer 1
    Dense(8, activation='relu'),               # Hidden layer 2
    Dense(1, activation='sigmoid')             # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")