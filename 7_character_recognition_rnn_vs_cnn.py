import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Conv2D, MaxPooling2D, Flatten
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

# RNN MODEL (Character Recognition)
# Treat image as sequence (28 time steps, 28 features)
rnn_model = Sequential([
    LSTM(128, input_shape=(28, 28)),   # better than SimpleRNN
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
rnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
print("\nTraining RNN Model...")
rnn_model.fit(X_train, y_train_cat,
              epochs=5,
              batch_size=128,
              validation_data=(X_test, y_test_cat))

# RNN PREDICTION (Character Recognition)
print("\nRNN Predictions:")
for i in range(5):
    sample = X_test[i]
    sample_input = sample.reshape(1, 28, 28)
    prediction = np.argmax(rnn_model.predict(sample_input, verbose=0))
    actual = y_test[i]
    print(f"Actual: {actual}  |  Predicted: {prediction}")

# CNN MODEL
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')])
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
print("\nTraining CNN Model...")
cnn_model.fit(X_train_cnn, y_train_cat,
              epochs=5,
              batch_size=128,
              validation_data=(X_test_cnn, y_test_cat))

# EVALUATION
rnn_acc = rnn_model.evaluate(X_test, y_test_cat, verbose=0)[1]
cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]
print("\n===== FINAL COMPARISON =====")
print(f"RNN Accuracy : {rnn_acc:.4f}")
print(f"CNN Accuracy : {cnn_acc:.4f}")

# VISUAL COMPARISON GRAPH
rnn_history = rnn_model.history.history
cnn_history = cnn_model.history.history
plt.plot(rnn_history['val_accuracy'], label='RNN')
plt.plot(cnn_history['val_accuracy'], label='CNN')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("RNN vs CNN Comparison")
plt.legend()
plt.show()

print("\n===== FINAL COMPARISON =====")
print(f"RNN Accuracy : {rnn_acc:.4f}")
print(f"CNN Accuracy : {cnn_acc:.4f}")
# WHICH MODEL IS BETTER
if cnn_acc > rnn_acc:
    print(f"\nConclusion: CNN is better by RNN{(cnn_acc - rnn_acc)*100:.2f}% accuracy.")
elif rnn_acc > cnn_acc:
    print(f"\nConclusion: RNN is better by CNN{(rnn_acc - cnn_acc)*100:.2f}% accuracy.")
else:
    print("\nConclusion: Both models have equal performance.")
