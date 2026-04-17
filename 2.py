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