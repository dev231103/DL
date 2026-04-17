import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Original X_train shape:", X_train.shape)  # (60000, 28, 28)
print("Original X_test shape:", X_test.shape)  # (10000, 28, 28)

# Normalize pixel values (0 to 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# -------------------------------
# 2. Prepare data for RNN
# -------------------------------
# RNN expects input as (samples, timesteps, features)
# Each image is 28 rows x 28 columns
# So treat each image as a sequence of 28 timesteps, each with 28 features
X_train_rnn = X_train.reshape(-1, 28, 28)
X_test_rnn = X_test.reshape(-1, 28, 28)

# -------------------------------
# 3. Build RNN Model
# -------------------------------
rnn_model = Sequential([
    SimpleRNN(128, activation='tanh', input_shape=(28, 28)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

rnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nRNN Model Summary:")
rnn_model.summary()

# -------------------------------
# 4. Train RNN Model
# -------------------------------
rnn_history = rnn_model.fit(
    X_train_rnn, y_train_cat,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
    verbose=1
)
# -------------------------------
# 5. Evaluate RNN Model
# -------------------------------
rnn_loss, rnn_acc = rnn_model.evaluate(X_test_rnn, y_test_cat, verbose=0)
print(f"\nRNN Test Accuracy: {rnn_acc * 100:.2f}%")

# -------------------------------
# 6. Prepare data for CNN
# -------------------------------
# CNN expects input as (samples, height, width, channels)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

# -------------------------------
# 7. Build CNN Model
# -------------------------------
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nCNN Model Summary:")
cnn_model.summary()

# -------------------------------
# 8. Train CNN Model
# -------------------------------
cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cat,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
    verbose=1
)

# -------------------------------
# 9. Evaluate CNN Model
# -------------------------------
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f"\nCNN Test Accuracy: {cnn_acc * 100:.2f}%")

# -------------------------------
# 10. Compare Results
# -------------------------------
print("\n========== Final Comparison ==========")
print(f"RNN Accuracy : {rnn_acc * 100:.2f}%")
print(f"CNN Accuracy : {cnn_acc * 100:.2f}%")

if cnn_acc > rnn_acc:
    print("CNN performs better for image-based character recognition.")
else:
    print("RNN performs better (uncommon for image tasks like MNIST).")

# -------------------------------
# 11. Plot Accuracy and Loss Graphs
# -------------------------------
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(rnn_history.history['accuracy'], label='RNN Train Acc')
plt.plot(rnn_history.history['val_accuracy'], label='RNN Val Acc')
plt.plot(cnn_history.history['accuracy'], label='CNN Train Acc')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(rnn_history.history['loss'], label='RNN Train Loss')
plt.plot(rnn_history.history['val_loss'], label='RNN Val Loss')
plt.plot(cnn_history.history['loss'], label='CNN Train Loss')
plt.plot(cnn_history.history['val_loss'], label='CNN Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()