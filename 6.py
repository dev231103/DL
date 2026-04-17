import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Load Fashion MNIST Dataset
# -------------------------------
(X_train, _), (X_test, _) = fashion_mnist.load_data()

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# -------------------------------
# 2. Normalize and Flatten Data
# -------------------------------
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))

print("Flattened Training Shape:", X_train.shape)
print("Flattened Testing Shape :", X_test.shape)

# -------------------------------
# 3. Build Autoencoder Model
# -------------------------------
input_layer = Input(shape=(784,))

# Encoder
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
bottleneck = Dense(16, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(bottleneck)
decoded = Dense(256, activation='relu')(decoded)
output_layer = Dense(784, activation='sigmoid')(decoded)

# Full Autoencoder
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Encoder Model (for compressed representation)
encoder = Model(inputs=input_layer, outputs=bottleneck)

# -------------------------------
# 4. Compile Model
# -------------------------------
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print("\nAutoencoder Model Summary:")
autoencoder.summary()

# -------------------------------
# 5. Train Model
# -------------------------------
history = autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)

# -------------------------------
# 6. Encode and Reconstruct Data
# -------------------------------
encoded_imgs = encoder.predict(X_test, verbose=0)
decoded_imgs = autoencoder.predict(X_test, verbose=0)

print("\nEncoded Data Shape:", encoded_imgs.shape)
print("Decoded Data Shape :", decoded_imgs.shape)

# -------------------------------
# 7. Plot Simple Loss Graph
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.title('Autoencoder Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 8. Show Original and Reconstructed Images
# -------------------------------
n = 5
plt.figure(figsize=(12, 5))

for i in range(n):
    # Original image
    plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed image
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Decoded")
    plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# 9. Display Encoded Features
# -------------------------------
print("\nEncoded Features of First Test Image:")
print(encoded_imgs[0])

# -------------------------------
# 10. Compression Analysis
# -------------------------------
original_size = 784
compressed_size = 16
compression_ratio = original_size / compressed_size

print("\n========== Compression Analysis ==========")
print(f"Original Size per Image   : {original_size} values")
print(f"Encoded Size per Image    : {compressed_size} values")
print(f"Compression Ratio         : {compression_ratio:.2f}:1")