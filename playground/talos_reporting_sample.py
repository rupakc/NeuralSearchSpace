import talos as ta
import pandas as pd
from talos.model.normalizers import lr_normalizer
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, Nadam
from keras.activations import softmax
from keras.losses import categorical_crossentropy, logcosh


x, y = ta.datasets.iris()


def iris_model(x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Dense(params['first_neuron'],
                    input_dim=x_train.shape[1],
                    activation='relu'))

    model.add(Dropout(params['dropout']))
    model.add(Dense(y_train.shape[1],
                    activation=params['last_activation']))

    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'],
                  metrics=['acc'])

    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_val, y_val])

    return out, model


p = {'lr': (0.1, 10, 10),
     'first_neuron':[4, 8, 16, 32, 64, 128],
     'batch_size': [2, 3, 4],
     'epochs': [200],
     'dropout': (0, 0.40, 10),
     'optimizer': [Adam, Nadam],
     'loss': ['categorical_crossentropy'],
     'last_activation': ['softmax'],
     'weight_regulizer': [None]}

h = ta.Scan(x, y, params=p,
            model=iris_model,
            dataset_name='iris',
            experiment_no='1',
            grid_downsample=.01)

# accessing the results data frame
print(h.data.head())

# accessing epoch entropy values for each round
print(h.peak_epochs_df)

# access the summary details
print(h.details)

# use Scan object as input
r = ta.Reporting(h)

# use filename as input
r = ta.Reporting('iris_1.csv')
# access the dataframe with the results
r.data.head(-3)
# get the number of rounds in the Scan
r.rounds()

# get the highest result ('val_acc' by default)
r.high()

# get the highest result for any metric
r.high('acc')

# get the round with the best result
r.rounds2high()

# get the best paramaters
r.best_params()

# get correlation for hyperparameters against a metric
r.correlate('val_loss')

# a regression plot for two dimensions
r.plot_regs()

# line plot
r.plot_line()

# up to two dimensional kernel density estimator
r.plot_kde('val_acc')

# a simple histogram
r.plot_hist(bins=50)

# heatmap correlation
r.plot_corr()

# a four dimensional bar grid
r.plot_bars('batch_size', 'val_acc', 'first_neuron', 'lr')

e = ta.Evaluate(h)
e.evaluate(x, y, folds=10, average='macro')

ta.Deploy(h, 'iris')

iris = ta.Restore('iris.zip')

# make predictions with the model
iris.model.predict(x)

# get the meta-data for the experiment
print(iris.details)
# get the hyperparameter space boundary
print(iris.params)
# sample of x and y data
print(iris.x)
print(iris.y)
# the results dataframe
print(iris.results)
