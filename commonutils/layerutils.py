from keras.layers import Dense, LSTM, GRU, Conv2D, Conv1D
from keras.models import Sequential
from constants import model_constants


def add_output_layer(model, output_dimension, default_activation='softmax', kernel_initializer='glorot_uniform',
                     bias_initializer='glorot_uniform', activity_regularizer='l1_l2'):
    model.add(Dense(output_dimension, activation=default_activation, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer, activity_regularizer=activity_regularizer))
    return model


def get_initialized_model_with_input_layer(number_of_units, input_shape, rnn_cell_type='lstm'):
    model = Sequential()
    if rnn_cell_type.lower() == model_constants.RNN_LSTM_MODEL_TYPE:
        model.add(LSTM(number_of_units, input_shape=input_shape))
    elif rnn_cell_type.lower() == model_constants.RNN_GRU_MODEL_TYPE:
        model.add(GRU(number_of_units, input_shape=input_shape))
    else:
        model.add(Dense(number_of_units, input_shape=input_shape)) # TODO - Add support for CNN
    return model
