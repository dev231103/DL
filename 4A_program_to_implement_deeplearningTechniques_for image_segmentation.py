# !pip install tensorflow matplotlib opencv-python
# !wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# !wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
# !tar -xzf images.tar.gz
# !tar -xzf annotations.tar.gz

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

IMG_SIZE = 128
# Load images
def load_data(img_path, mask_path, limit=1000):
    images = []
    masks = []
    img_files = os.listdir(img_path)[:limit]
    for file in img_files:
        img_full_path = os.path.join(img_path, file)
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Could not read image file: {img_full_path}. Skipping.")
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # The mask files are .png and located in annotations/trimaps
        mask_file = file.replace('.jpg', '.png')
        mask_full_path = os.path.join(mask_path, mask_file)
        mask = cv2.imread(mask_full_path, 0) # Read mask in grayscale
        if mask is None:
            print(f"Warning: Could not read mask file: {mask_full_path}. Skipping corresponding image.")
            continue # If mask is missing, skip the image too
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        images.append(img)
        masks.append(mask)
    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0
    masks = np.expand_dims(masks, axis=-1)
    return images, masks
X, Y = load_data("images", "annotations/trimaps")

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# U-Net model
def unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)
    model = models.Model(inputs, outputs)
    return model
model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, Y_train, epochs=30, batch_size=8)
# Predict
preds = model.predict(X_test)

# Display result
plt.subplot(1,3,1)
plt.imshow(X_test[0])
plt.title("Input")
plt.subplot(1,3,2)
plt.imshow(Y_test[0].squeeze(), cmap='gray')
plt.title("Actual Mask")
plt.subplot(1,3,3)
plt.imshow(preds[0].squeeze(), cmap='gray')
plt.title("Predicted Mask")
plt.show()
