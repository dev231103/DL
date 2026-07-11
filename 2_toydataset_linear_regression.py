
# PART 1: DIABETES CLASSIFICATION MODEL

from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Load dataset (Diabetes dataset)
dataset = loadtxt('/content/sample_data/diabetes (1).csv', delimiter=',', skiprows=1)

# Split dataset into input (X) and output (Y)
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Display sample data
print("Sample Input Data:\n", X[:5])
print("\nSample Output Data:\n", Y[:5])

# Create Neural Network model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))   # Hidden layer 1
model.add(Dense(8, activation='relu'))                 # Hidden layer 2
model.add(Dense(1, activation='sigmoid'))              # Output layer

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

# Evaluate model performance
_, accuracy = model.evaluate(X, Y, verbose=0)
print("\nAccuracy of model is:", accuracy * 100)

# Make predictions (convert probability → 0 or 1)
predictions = model.predict(X)
prediction = (predictions > 0.5).astype(int)

# Display first 5 results
print("\nFirst 5 Predictions (Classification):\n")
for i in range(5):
    print("Input:", X[i].tolist(),
          "Predicted:", prediction[i][0],
          "Actual:", int(Y[i])


#part 2


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([500, 800, 1000, 1200, 1500, 1800, 2000, 2200], dtype=np.float32)
Y = np.array([150, 200, 250, 280, 350, 400, 450, 500], dtype=np.float32)

X = X / 1000.0
Y = Y / 1000.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mse')
history = model.fit(X, Y, epochs=200, verbose=0)

plt.plot(history.history['loss'])
plt.title("Loss Function")
plt.show()

predicted = model.predict(X)

plt.scatter(X * 1000, Y * 1000)
plt.plot(X * 1000, predicted * 1000)
plt.title("Linear Regression")
plt.show()

new_data = np.array([1300, 2500], dtype=np.float32) / 1000.0
predictions = model.predict(new_data)

for sqft, price in zip([1300, 2500], predictions):
    print(f"Predicted price for {sqft} sq ft = {price[0] * 1000:.2f}")
