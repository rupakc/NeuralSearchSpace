OPTIMIZER_LIST = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
ACTIVATION_FUNCTION_LIST = ['softmax','elu','selu', 'softplus', 'softsign','relu','tanh', 'sigmoid', 'hard_sigmoid', 'linear']
REGRESSION_LOSS_FUNCTION_LIST = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error',
                                 'squared_hinge', 'hinge', 'logcosh']
CLASSIFICATION_LOSS_FUNCTION_LIST = ['categorical_hinge','categorical_crossentropy','sparse_categorical_crossentropy',
                                     'binary_crossentropy']
OTHER_LOSS_FUNCTION_LIST = ['kullback_leibler_divergence', 'poisson', 'cosine_proximity']
WEIGHT_INITIALIZER_LIST = ['lecun_uniform','glorot_normal','glorot_uniform','he_normal','lecun_normal','he_uniform','random_uniform','random_normal']
IMAGE_DATASET_LIST = ['cifar10','cifar100', 'mnist', 'fashion_mnist']
TEXT_DATASET_LIST = ['imdb','reuters']
REGRESSION_DATASET_LIST = ['boston_housing']
REGULARIZER_LIST = ['l1','l2','l1_l2']
POOLING_TYPE_LIST = ['max', 'average']
CLASSIFICATION_MODEL_TYPE = 'classification'
REGRESSION_MODEL_TYPE = 'regression'
DNN_MODEL_TYPE = 'dnn'
CNN_MODEL_TYPE = 'cnn'
RNN_LSTM_MODEL_TYPE = 'lstm'
RNN_GRU_MODEL_TYPE = 'gru'
LOG_FILE_PATH = ''
CNN_TYPE_1D = '1D'
CNN_TYPE_2D = '2D'
CNN_POOLING_MAX = 'max'
CNN_POOLING_AVERAGE = 'average'

OPTIMIZER_TYPE = 'optimizer'
LOSS_TYPE = 'loss'
INITIALIZER_TYPE = 'initializer'
ACTIVATION_TYPE = 'activation'
REGULARIZER_TYPE = 'regularizer'



