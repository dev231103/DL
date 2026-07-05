import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# LOAD DATA
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# ==========================================
# RNN MODEL (Character Recognition)
# ==========================================

rnn_model = Sequential([
    Input(shape=(28, 28)),
    LSTM(128),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

rnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining RNN Model...")

rnn_history = rnn_model.fit(
    X_train,
    y_train_cat,
    epochs=5,
    batch_size=128,
    validation_data=(X_test, y_test_cat)
)

# ==========================================
# RNN PREDICTION
# ==========================================

print("\nRNN Predictions:")

for i in range(5):
    sample = X_test[i]
    sample_input = sample.reshape(1, 28, 28)

    prediction = np.argmax(
        rnn_model.predict(sample_input, verbose=0)
    )

    actual = y_test[i]

    print(f"Actual: {actual}  |  Predicted: {prediction}")

# ==========================================
# CNN MODEL
# ==========================================

X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

cnn_model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN Model...")

cnn_history = cnn_model.fit(
    X_train_cnn,
    y_train_cat,
    epochs=5,
    batch_size=128,
    validation_data=(X_test_cnn, y_test_cat)
)

# ==========================================
# EVALUATION
# ==========================================

rnn_acc = rnn_model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)[1]

cnn_acc = cnn_model.evaluate(
    X_test_cnn,
    y_test_cat,
    verbose=0
)[1]

print("\n===== FINAL COMPARISON =====")
print(f"RNN Accuracy : {rnn_acc:.4f}")
print(f"CNN Accuracy : {cnn_acc:.4f}")

# ==========================================
# VISUAL COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(rnn_history.history['val_accuracy'], label='RNN')
plt.plot(cnn_history.history['val_accuracy'], label='CNN')

plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("RNN vs CNN Comparison")
plt.legend()
plt.grid(True)

plt.show()

# ==========================================
# FINAL COMPARISON
# ==========================================

print("\n===== FINAL COMPARISON =====")
print(f"RNN Accuracy : {rnn_acc:.4f}")
print(f"CNN Accuracy : {cnn_acc:.4f}")

# ==========================================
# CONCLUSION
# ==========================================

if cnn_acc > rnn_acc:
    print(f"\nConclusion: CNN is better than RNN by {(cnn_acc - rnn_acc) * 100:.2f}% accuracy.")
elif rnn_acc > cnn_acc:
    print(f"\nConclusion: RNN is better than CNN by {(rnn_acc - cnn_acc) * 100:.2f}% accuracy.")
else:
    print("\nConclusion: Both models have equal performance.")
