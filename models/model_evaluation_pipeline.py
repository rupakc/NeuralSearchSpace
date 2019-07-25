'''
1. Get the dataset
2. Re-format the dataset as per the network type
3. Initialize the network config hyperparameters
4. Get the model
5. Train the model
6. Measure the performance of the model
7. Persist the results in a database
----------------------------------------
'''

from commonutils import datautils, logutils, dbutils
from commonutils.customexception import NeuralConfigError
from models import generator_factory


class BenchmarkModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_dataset(self):
        X_train, y_train, X_test, y_test = datautils.get_dataset_from_name(self.dataset_name)
        return X_train, y_train, X_test, y_test

    def format_dataset(self): # TODO - Format the dataset as per the network-type
        pass

    def initialize_network_config(self, input_shape, input_units, output_units, number_of_layers, output_metric_name_list,
                 nodes_per_layer_list, activation_function_name, kernel_initialization_function_name, output_layer_activation,
                 bias_initialization_function_name, activity_regularizer_name,
                 constraint_name, optimizer_name, loss_name, output_loss_type, network_type, rnn_cell_type): # TODO - Configure the network with parameters and hyperparameters

        builder_factory = generator_factory.NetworkBuilderFactory(input_shape, input_units, output_units, number_of_layers, output_metric_name_list,
                 nodes_per_layer_list, activation_function_name, kernel_initialization_function_name, output_layer_activation,
                 bias_initialization_function_name, activity_regularizer_name,
                 constraint_name, optimizer_name, loss_name, output_loss_type, network_type, rnn_cell_type)

        self.builder_factory = builder_factory

    def get_model_or_model_iterator(self, iterating_variable):
        if iterating_variable is None:
            model = self.builder_factory.build_network(iterating_variable)
            return [model]
        model_generator = self.builder_factory.build_network(iterating_variable)
        return model_generator

    def train_model(self, model_or_model_generator_list):
        for model in model_or_model_generator_list:
            model.fit()
        pass

    def evaluate_model(self):
        pass

    def persist_results(self, network_object_dict):
        dbutils.check_for_duplicate_and_insert(network_object_dict)

    def execution_pipeline(self):
        pass
