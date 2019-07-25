from constants import model_constants
from keras.layers import Dense, LSTM, GRU, Conv2D, Conv1D, MaxPooling1D, AveragePooling2D, MaxPooling2D, AveragePooling1D, Flatten
from copy import deepcopy
from commonutils.customexception import NeuralConfigError


def get_compiled_model_generator_with_optimizer(model, loss_function_name, metrics_list):
    for optimizer_name in model_constants.OPTIMIZER_LIST:
        model.compile(optimizer=optimizer_name,
                      loss=loss_function_name,
                      metrics=metrics_list)
        yield model


def get_compiled_model_generator_with_loss_function(model, optimizer_name, metrics_list, loss_type='regression'):
    if loss_type.lower() == model_constants.REGRESSION_MODEL_TYPE:
        for loss_function_name in model_constants.REGRESSION_LOSS_FUNCTION_LIST:
            model.compile(optimizer=optimizer_name,
                          loss=loss_function_name,
                          metrics=metrics_list)
            yield model
    elif loss_type.lower() == model_constants.CLASSIFICATION_MODEL_TYPE:
        for loss_function_name in model_constants.CLASSIFICATION_LOSS_FUNCTION_LIST:
            model.compile(optimizer=optimizer_name,
                          loss=loss_function_name,
                          metrics=metrics_list)
            yield model
    else:
        for loss_function_name in model_constants.OTHER_LOSS_FUNCTION_LIST:
            model.compile(optimizer=optimizer_name,
                          loss=loss_function_name,
                          metrics=metrics_list)
            yield model


def get_dnn_block(model, num_hidden_layers, node_per_layer_list, activation_function='relu',
                  kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activity_regularizer='l1_l2'):
    for index, _ in enumerate(range(num_hidden_layers)):
        model.add(Dense(node_per_layer_list[index], activation=activation_function,
                        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                        activity_regularizer=activity_regularizer))
    return model


def get_rnn_block(model, num_hidden_layers, node_per_layer_list, rnn_cell_type='lstm', activation_function='tanh',
                  kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activity_regularizer='l1_l2'):
    if rnn_cell_type.lower() == model_constants.RNN_LSTM_MODEL_TYPE:
        for index, _ in enumerate(range(num_hidden_layers-1)):
            model.add(LSTM(node_per_layer_list[index], return_sequences=True, stateful=True,
                           activation=activation_function, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                           activity_regularizer=activity_regularizer, recurrent_dropout=0.2,dropout=0.2))
        model.add(LSTM(node_per_layer_list[len(node_per_layer_list)-1], return_sequences=False, stateful=True,
                           activation=activation_function, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                           activity_regularizer=activity_regularizer, recurrent_dropout=0.2, dropout=0.2))
    elif rnn_cell_type.lower() == model_constants.RNN_GRU_MODEL_TYPE:
        for index, _ in enumerate(range(num_hidden_layers-1)):
            model.add(GRU(node_per_layer_list[index], return_sequences=True, stateful=True,
                           activation=activation_function, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                           activity_regularizer=activity_regularizer, recurrent_dropout=0.2,dropout=0.2))
        model.add(GRU(node_per_layer_list[len(node_per_layer_list)-1], return_sequences=False, stateful=True,
                           activation=activation_function, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                           activity_regularizer=activity_regularizer, recurrent_dropout=0.2, dropout=0.2))
    else:
        raise NeuralConfigError("Model-type not supported or ill-configured")
    return model


def get_cnn_block(model, num_hidden_layers, conv_layer_per_block, conv_filter_vector, kernel_size_vector, pool_size_vector, pooling_type = 'max', conv_stride=1, pool_stride=1, padding='same', conv_type='2D', activation_function='relu',
                  kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activity_regularizer='l1_l2'):
    if num_hidden_layers%(conv_layer_per_block+1) == 0:
        number_of_blocks = num_hidden_layers/(conv_layer_per_block+1)
        for block_index, _ in enumerate(range(number_of_blocks)):
            for conv_layer_index, _ in enumerate(range(conv_layer_per_block)):
                if conv_type.lower() == model_constants.CNN_TYPE_1D:
                    model.add(Conv1D(filters=conv_filter_vector[conv_layer_index],kernel_size=kernel_size_vector[conv_layer_index], strides=conv_stride,
                                     padding=padding, activation=activation_function,
                                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                     activity_regularizer=activity_regularizer))
                elif conv_type.lower() == model_constants.CNN_TYPE_2D:
                    model.add(Conv2D(filters=conv_filter_vector[conv_layer_index],kernel_size=kernel_size_vector[conv_layer_index], strides=conv_stride,
                                     padding=padding, activation=activation_function,
                                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                     activity_regularizer=activity_regularizer))
                else:
                    raise NeuralConfigError("Model-type not supported or ill-configured")

            if pooling_type.lower() == model_constants.CNN_POOLING_MAX:
                if conv_type.lower() == model_constants.CNN_TYPE_1D:
                    model.add(MaxPooling1D(pool_size=pool_size_vector[block_index], strides=pool_stride, activation=activation_function, kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     activity_regularizer=activity_regularizer))
                elif conv_type.lower() == model_constants.CNN_TYPE_2D:
                    model.add(MaxPooling2D(pool_size=pool_size_vector[block_index], strides=pool_stride, activation=activation_function, kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     activity_regularizer=activity_regularizer))
            elif pooling_type.lower() == model_constants.CNN_POOLING_AVERAGE:
                if conv_type.lower() == model_constants.CNN_TYPE_1D:
                    model.add(AveragePooling1D(pool_size=pool_size_vector[block_index], strides=pool_stride, activation=activation_function, kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     activity_regularizer=activity_regularizer))
                elif conv_type.lower() == model_constants.CNN_TYPE_2D:
                    model.add(AveragePooling2D(pool_size=pool_size_vector[block_index], strides=pool_stride, activation=activation_function, kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     activity_regularizer=activity_regularizer))
        model.add(Flatten())
    else:
        raise NeuralConfigError("Model ill-configured. Layer configuration is incompatible.")
    return model


def get_generator_over_initializer(model, num_hidden_layers, node_per_layer_list, activation_function='relu',
                                   activity_regularizer='l1_l2', model_type='dnn', conv_layer_per_block= [], conv_filter_vector= [],
                                   kernel_size_vector= [], pool_size_vector= [], pooling_type= 'max', conv_stride= 1, pool_stride= 1,
                                   padding= 'same', conv_type='2D'):
    for initializer_name in model_constants.WEIGHT_INITIALIZER_LIST:
        copy_model = deepcopy(model)
        if model_type.lower() == model_constants.DNN_MODEL_TYPE:
            structured_model = get_dnn_block(copy_model, num_hidden_layers, node_per_layer_list, activation_function,
                                  kernel_initializer=initializer_name, bias_initializer=initializer_name,
                                  activity_regularizer=activity_regularizer)
            yield structured_model
        if model_type.lower() == model_constants.RNN_GRU_MODEL_TYPE or model_type.lower() == model_constants.RNN_LSTM_MODEL_TYPE:
           structured_model = get_rnn_block(copy_model, num_hidden_layers, node_per_layer_list, rnn_cell_type=model_type,
                                            kernel_initializer=initializer_name, bias_initializer=initializer_name,
                                            activity_regularizer=activity_regularizer)
           yield structured_model
        if model_type.lower() == model_constants.CNN_MODEL_TYPE:
           structured_model = get_cnn_block(copy_model, num_hidden_layers, conv_layer_per_block, conv_filter_vector,
                                            kernel_size_vector, pool_size_vector, pooling_type=pooling_type,
                                            conv_stride=conv_stride,
                                            pool_stride=pool_stride, padding=padding, conv_type=conv_type,
                                            activation_function=activation_function,
                                            kernel_initializer=initializer_name,
                                            bias_initializer=initializer_name,
                                            activity_regularizer=activity_regularizer)
           yield structured_model


def get_generator_over_activations(model, num_hidden_layers, node_per_layer_list= [],
                                   kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform',
                                   activity_regularizer='l1_l2', model_type='dnn', conv_layer_per_block= [], conv_filter_vector= [],
                                   kernel_size_vector= [], pool_size_vector= [], pooling_type= 'max', conv_stride= 1, pool_stride= 1,
                                   padding= 'same', conv_type='2D'):
    for activation_function_name in model_constants.ACTIVATION_FUNCTION_LIST:
        copy_model = deepcopy(model)
        if model_type.lower() == model_constants.DNN_MODEL_TYPE:
            structured_model = get_dnn_block(copy_model, num_hidden_layers, node_per_layer_list,
                                  activation_function=activation_function_name,
                                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                  activity_regularizer=activity_regularizer)
            yield structured_model
        if model_type.lower() == model_constants.RNN_LSTM_MODEL_TYPE or model_type.lower() == model_constants.RNN_GRU_MODEL_TYPE:
            structured_model = get_rnn_block(copy_model, num_hidden_layers, node_per_layer_list, rnn_cell_type=model_type,
                                             activation_function=activation_function_name,
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                             activity_regularizer=activity_regularizer)
            yield structured_model
        if model_type.lower() == model_constants.CNN_MODEL_TYPE:
            structured_model = get_cnn_block(copy_model, num_hidden_layers, conv_layer_per_block, conv_filter_vector,
                                             kernel_size_vector, pool_size_vector, pooling_type=pooling_type, conv_stride=conv_stride,
                                             pool_stride=pool_stride, padding=padding, conv_type=conv_type, activation_function=activation_function_name,
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                             activity_regularizer=activity_regularizer)
            yield structured_model


def get_generator_over_regularizers(model, num_hidden_layers, node_per_layer_list, activation_function='relu',
                                    kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', model_type='dnn', conv_layer_per_block= [], conv_filter_vector= [],
                                   kernel_size_vector= [], pool_size_vector= [], pooling_type= 'max', conv_stride= 1, pool_stride= 1,
                                   padding= 'same', conv_type='2D'):
    for activity_regularizer_name in model_constants.REGULARIZER_LIST:
        copy_model = deepcopy(model)
        if model_type.lower() == model_constants.DNN_MODEL_TYPE:
            structured_model = get_dnn_block(copy_model, num_hidden_layers, node_per_layer_list,
                                  activation_function=activation_function,
                                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                  activity_regularizer=activity_regularizer_name)
            yield structured_model
        if model_type.lower() == model_constants.RNN_GRU_MODEL_TYPE or model_type.lower() == model_constants.RNN_LSTM_MODEL_TYPE:
            structured_model = get_rnn_block(copy_model, num_hidden_layers, node_per_layer_list, rnn_cell_type=model_type,
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                             activity_regularizer=activity_regularizer_name)
            yield structured_model
        if model_type.lower() == model_constants.CNN_MODEL_TYPE:
            structured_model = get_cnn_block(copy_model, num_hidden_layers, conv_layer_per_block, conv_filter_vector,
                                             kernel_size_vector, pool_size_vector, pooling_type=pooling_type, conv_stride=conv_stride,
                                             pool_stride=pool_stride, padding=padding, conv_type=conv_type, activation_function=activation_function,
                                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                             activity_regularizer=activity_regularizer_name)
            yield structured_model


def get_generator_over_pooling(model, num_hidden_layers, conv_layer_per_block, conv_filter_vector, kernel_size_vector, pool_size_vector, conv_stride = 1, pool_stride = 1, padding = 'valid', conv_type = '2D',  activation_function='relu',
                                    kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activity_regularizer='l1_l2', model_type='cnn'):
    for pooling_name in model_constants.POOLING_TYPE_LIST:
        copy_model = deepcopy(model)
        if model_type.lower() == model_constants.CNN_MODEL_TYPE:
            structured_model = get_cnn_block(copy_model, num_hidden_layers, conv_layer_per_block, conv_filter_vector, kernel_size_vector= kernel_size_vector, pool_size_vector= pool_size_vector, pooling_type=pooling_name, conv_stride = conv_stride, pool_stride= pool_stride, padding= padding, conv_type = conv_type,
                                  activation_function=activation_function,
                                  kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                  activity_regularizer=activity_regularizer)
            yield structured_model
        else:
            raise NeuralConfigError("Pooling valid only for Convolutional Neural Networks")


# from keras.models import Sequential
# from keras.layers import Dense
# model = Sequential()
# model.add(LSTM(10, input_shape=(20, 300), batch_input_shape=(32,20,300),return_sequences=True, stateful=True))
#
# #model = get_dnn_block(model,5,[10,10,20,30,40])
#
# for m in get_generator_over_regularizers(model, 5, [10, 10, 20, 30, 40], model_type='lstm', activation_function='tanh'):
#     m.summary()
#     # model = Sequential()
#     # model.add(Dense(10, input_dim=20))
