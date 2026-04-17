from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, Y, epochs=500, verbose=0)
Xnew, Yreal = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scaler.transform(Xnew)
Ynew = model.predict(Xnew, verbose=0)
Yclass = (Ynew > 0.5).astype("int32")

for i in range(len(Xnew)):
    print("X=%s, Predicted_probability=%s, Predicted_class=%s" % (Xnew[i], Ynew[i][0], Yclass[i][0]))











    #REGRESSION
    # from keras.models import Sequential
    # from keras.layers import Dense
    # from sklearn.datasets import make_regression
    # from sklearn.preprocessing import MinMaxScaler
    #
    # X, Y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)
    # scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    # scalarX.fit(X)
    # scalarY.fit(Y.reshape(100, 1))
    # X = scalarX.transform(X)
    # Y = scalarY.transform(Y.reshape(100, 1))
    #
    # model = Sequential()
    # model.add(Dense(4, input_dim=2, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='mse', optimizer='adam')
    # model.fit(X, Y, epochs=1000, verbose=0)
    #
    # Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
    # Xnew = scalarX.transform(Xnew)
    # Ynew = model.predict(Xnew)
    #
    # for i in range(len(Xnew)):
    #     print("X=%s,Predicted=%s" % (Xnew[i], Ynew[i]))