import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Load and Prepare MNIST Dataset
# -------------------------------
(X_train, _), (_, _) = mnist.load_data()

# Normalize images to [-1, 1]
X_train = X_train.astype("float32") / 127.5 - 1.0
X_train = np.expand_dims(X_train, axis=-1)

print("Training Data Shape:", X_train.shape)

img_shape = (28, 28, 1)
latent_dim = 100

# -------------------------------
# 2. Build Generator
# -------------------------------
def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(784, activation='tanh'))
    model.add(Reshape(img_shape))
    return model

# -------------------------------
# 3. Build Discriminator
# -------------------------------
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Create Generator and Discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile Discriminator
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy']
)

# -------------------------------
# 4. Build GAN Model
# -------------------------------
z = Input(shape=(latent_dim,))
generated_img = generator(z)

discriminator.trainable = False
validity = discriminator(generated_img)

gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

print("\nGenerator Summary:")
generator.summary()

print("\nDiscriminator Summary:")
discriminator.summary()

# -------------------------------
# 5. Training Parameters
# -------------------------------
epochs = 5000
batch_size = 64
sample_interval = 1000

d_losses = []
g_losses = []

# -------------------------------
# 6. Function to Save Generated Images
# -------------------------------
def sample_images(epoch):
    noise = np.random.normal(0, 1, (16, latent_dim))
    gen_imgs = generator.predict(noise, verbose=0)

    # Rescale images from [-1,1] to [0,1]
    gen_imgs = 0.5 * gen_imgs + 0.5

    plt.figure(figsize=(6, 6))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(gen_imgs[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.suptitle(f"Generated Images at Epoch {epoch}")
    plt.tight_layout()
    plt.show()

# -------------------------------
# 7. Train GAN
# -------------------------------
for epoch in range(epochs):

    # -------------------
    # Train Discriminator
    # -------------------
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise, verbose=0)

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # -------------------
    # Train Generator
    # -------------------
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.ones((batch_size, 1))

    g_loss = gan.train_on_batch(noise, valid_y)

    # Store losses
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)

    # Print progress
    if epoch % 500 == 0:
        print(f"Epoch {epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

    # Show generated images
    if epoch % sample_interval == 0:
        sample_images(epoch)

# -------------------------------
# 8. Plot Loss Graph
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('GAN Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 9. Final Generated Images
# -------------------------------
print("\nFinal Generated Images:")
sample_images(epochs)