# Practical 6: Applying Autoencoder algorithms for encoding real-world data
# Dataset used: MNIST handwritten digit images

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Step 1: Load Real-World Dataset
# -------------------------------
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten 28x28 images into 784-dimensional vectors
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# -------------------------------
# Step 2: Define Autoencoder Model
# -------------------------------
input_dim = 784      # Original image size
encoding_dim = 32    # Compressed representation size

# Input layer
input_img = Input(shape=(input_dim,))

# Encoder
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded_output = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded_output)
decoded = Dense(128, activation='relu')(decoded)
decoded_output = Dense(input_dim, activation='sigmoid')(decoded)

# Full autoencoder model
autoencoder = Model(input_img, decoded_output)

# Encoder model (for extracting compressed features)
encoder = Model(input_img, encoded_output)

# -------------------------------
# Step 3: Build Decoder Separately
# -------------------------------
encoded_input = Input(shape=(encoding_dim,))

decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_layer3 = autoencoder.layers[-1](decoder_layer2)

decoder = Model(encoded_input, decoder_layer3)

# -------------------------------
# Step 4: Compile the Model
# -------------------------------
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

# -------------------------------
# Step 5: Train the Autoencoder
# -------------------------------
history = autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# -------------------------------
# Step 6: Encode and Decode Test Images
# -------------------------------
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

print("Encoded representation shape:", encoded_imgs.shape)

# -------------------------------
# Step 7: Plot Training Loss
# -------------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Autoencoder Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Step 8: Visualize Original vs Reconstructed Images
# -------------------------------
n = 10  # Number of images to display
plt.figure(figsize=(20, 4))

for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------
# Step 9: Display Encoded Data for One Sample
# -------------------------------
sample_index = 0
print("\nCompressed (Encoded) Feature Vector for Sample", sample_index)
print(encoded_imgs[sample_index])

# -------------------------------
# Step 10: Compression Summary
# -------------------------------
original_size = input_dim
compressed_size = encoding_dim
compression_ratio = original_size / compressed_size

print("\nOriginal Size:", original_size)
print("Compressed Size:", compressed_size)
print("Compression Ratio: {:.2f}:1".format(compression_ratio))
