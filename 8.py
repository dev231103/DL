import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Load MNIST Dataset
# -------------------------------
(X_train, _), (X_test, _) = mnist.load_data()

print("Original Training Data Shape:", X_train.shape)
print("Original Testing Data Shape :", X_test.shape)

# -------------------------------
# 2. Normalize and Flatten Data
# -------------------------------
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Flatten 28x28 images into 784-dimensional vectors
X_train = X_train.reshape((len(X_train), 784))
X_test = X_test.reshape((len(X_test), 784))

print("Flattened Training Data Shape:", X_train.shape)
print("Flattened Testing Data Shape :", X_test.shape)

# -------------------------------
# 3. Define Autoencoder Architecture
# -------------------------------
input_dim = 784
encoding_dim = 32   # compressed representation

# Input Layer
input_img = Input(shape=(input_dim,))

# Encoder
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded_output = Dense(encoding_dim, activation='relu', name='encoded_layer')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded_output)
decoded = Dense(128, activation='relu')(decoded)
decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder Model
autoencoder = Model(input_img, decoded_output)

# Separate Encoder Model
encoder = Model(input_img, encoded_output)

# -------------------------------
# 4. Compile Model
# -------------------------------
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

print("\nAutoencoder Model Summary:")
autoencoder.summary()

# -------------------------------
# 5. Train the Autoencoder
# -------------------------------
history = autoencoder.fit(
    X_train, X_train,                  # input = output
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=1
)

# -------------------------------
# 6. Encode and Reconstruct Test Data
# -------------------------------
encoded_imgs = encoder.predict(X_test, verbose=0)
decoded_imgs = autoencoder.predict(X_test, verbose=0)

print("\nEncoded Data Shape:", encoded_imgs.shape)
print("Reconstructed Data Shape:", decoded_imgs.shape)

# -------------------------------
# 7. Plot Training and Validation Loss
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='s', label='Validation Loss')
plt.title('Autoencoder Training vs Validation Loss (MNIST)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 8. Display Original vs Reconstructed Images
# -------------------------------
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # Original images
    plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed images
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Decoded")
    plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# 9. Display Encoded Features of First Sample
# -------------------------------
print("\nEncoded Representation of First Test Image (32 features):")
print(encoded_imgs[0])

# -------------------------------
# 10. Compression Analysis
# -------------------------------
original_size = 784
compressed_size = encoding_dim
compression_ratio = original_size / compressed_size

print("\n========== Compression Analysis ==========")
print(f"Original Size per Image    : {original_size} values")
print(f"Compressed Size per Image  : {compressed_size} values")
print(f"Compression Ratio          : {compression_ratio:.2f}:1")