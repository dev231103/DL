# !pip install tensorflow scikit-learn numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Create dataset with 3 classes (IMPORTANT CHANGE)
X, Y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=1)

# Normalize data
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

# Create model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))

# Output layer for MULTICLASS (IMPORTANT CHANGE)
model.add(Dense(3, activation='softmax'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train model
model.fit(X, Y, epochs=50, verbose=0)

# New data for testing
Xnew, Yreal = make_blobs(n_samples=3, centers=3, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)

# Predict class
Ypred = np.argmax(model.predict(Xnew), axis=-1)

# Print results
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s, Actual=%s" % (Xnew[i], Ypred[i], Yreal[i]))
