from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset = loadtxt('/content/sample_data/diabetes.csv', delimiter=',', skiprows=1)
X = dataset[:, 0:8]
Y = dataset[:, 8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=150, batch_size=10)
loss, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy of model is', accuracy * 100)
prediction = model.predict(X_test)
for i in range(5):
    print(X_test[i].tolist(), prediction[i], Y_test[i])