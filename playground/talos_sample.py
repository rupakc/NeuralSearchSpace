from numpy import loadtxt
from keras.activations import relu, elu
from keras.models import Sequential
from keras.layers import Dense
import talos as ta


dataset = loadtxt("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
                  delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]
p={
    'first_neuron': [12, 24, 48],
    'activation': [relu, elu],
    'batch_size': [10, 20, 30]
}


def diabetes(x_train, y_train, x_val, y_val, params):
    # replace the hyperparameter inputs with references to params dictionary
    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=8, activation=params['activation']))
    # model.add(Dense(8, activation=params['activation']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # make sure history object is returned by model.fit()
    out = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=params['batch_size'],
                    # validation_split=.3,
                    verbose=0)

    # modify the output model
    return out, model


t = ta.Scan(X, Y, p, diabetes)
print(t)

