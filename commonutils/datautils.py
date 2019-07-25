from keras.datasets import boston_housing, cifar10, cifar100
from keras.datasets import fashion_mnist, imdb, mnist, reuters
from commonutils.customexception import NeuralConfigError


def get_dataset_from_name(dataset_name):
    dataset_name_object_dict = {'cifar10': cifar10, 'cifar100': cifar100,
                                'mnist': mnist, 'fashion_mnist': fashion_mnist,
                                'imdb': imdb, 'reuters': reuters, 'boston_housing': boston_housing}

    if dataset_name.lower() in dataset_name_object_dict.keys():
        (X_train, y_train), (X_test, y_test) = dataset_name_object_dict[dataset_name.lower()].load_data()
        return X_train, y_train, X_test, y_test
    else:
        raise NeuralConfigError("Dataset misspelled or not supported")
