from models import generator_helper
from commonutils import layerutils, customexception
from constants import model_constants


class NetworkBuilderFactory:
    def __init__(self, input_shape, input_units, output_units, number_of_layers, output_metric_name_list,
                 nodes_per_layer_list, activation_function_name, kernel_initialization_function_name, output_layer_activation,
                 bias_initialization_function_name, activity_regularizer_name,
                 constraint_name, optimizer_name, loss_name, output_loss_type, network_type, rnn_cell_type):

        self.input_shape = input_shape
        self.input_units = input_units
        self.output_units = output_units
        self.number_of_layers = number_of_layers
        self.nodes_per_layer_list = nodes_per_layer_list
        self.activation_function_name = activation_function_name
        self.kernel_initialization_function_name = kernel_initialization_function_name
        self.bias_initialization_function_name = bias_initialization_function_name
        self.activity_regularizer_name = activity_regularizer_name
        self.constraint_name = constraint_name
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.output_layer_activation = output_layer_activation
        self.output_metric_name_list = output_metric_name_list
        self.output_loss_type = output_loss_type
        self.rnn_cell_type = rnn_cell_type
        self.network_type = network_type

    def build_network(self, iterating_variable):
        if self.network_type.lower() == model_constants.RNN_LSTM_MODEL_TYPE or self.network_type.lower() == model_constants.RNN_GRU_MODEL_TYPE:
            return self.build_rnn_network(self.network_type, iterating_variable)
        elif self.network_type.lower() == model_constants.CNN_MODEL_TYPE:
            pass  # TODO - Add self.build_cnn_network here
        elif self.network_type.lower() == model_constants.DNN_MODEL_TYPE:
            return self.build_dnn_network(iterating_variable)
        else:
            raise customexception.NeuralConfigError("Network type not supported or misspelled")

    def build_rnn_network(self, network_type, iterating_variable):
        model = layerutils.get_initialized_model_with_input_layer(self.input_units, self.input_shape) # TODO - Add RNN support in this function
        if iterating_variable is None:
            final_model = self.build_rnn_network_helper(model)
            final_model.compile(optimizer=self.optimizer_name,
                                loss=self.loss_name,
                                metrics=self.output_metric_name_list)
            return final_model

        elif iterating_variable.lower() == model_constants.OPTIMIZER_TYPE:
            final_model = self.build_rnn_network_helper(model)
            return generator_helper.get_compiled_model_generator_with_optimizer(final_model,
                                                                                self.loss_name,
                                                                                self.output_metric_name_list)

        elif iterating_variable.lower() == model_constants.LOSS_TYPE:
            final_model = self.build_rnn_network_helper(model)
            return generator_helper.get_compiled_model_generator_with_loss_function(final_model,
                                                                                    self.optimizer_name,
                                                                                    self.output_metric_name_list,
                                                                                    self.output_loss_type)

        elif iterating_variable.lower() == model_constants.INITIALIZER_TYPE:
            partial_model_generator = generator_helper.get_generator_over_initializer(model, self.number_of_layers,
                                                                                      self.nodes_per_layer_list,
                                                                                      self.activation_function_name,
                                                                                      self.activity_regularizer_name,
                                                                                      network_type)
            return self.build_rnn_network_generator_helper(partial_model_generator)

        elif iterating_variable.lower() == model_constants.ACTIVATION_TYPE:
            partial_model_generator = generator_helper.get_generator_over_activations(model, self.number_of_layers,
                                                                                      self.nodes_per_layer_list,
                                                                                      self.kernel_initialization_function_name,
                                                                                      self.bias_initialization_function_name,
                                                                                      self.activity_regularizer_name,
                                                                                      network_type)
            return self.build_rnn_network_generator_helper(partial_model_generator)

        elif iterating_variable.lower() == model_constants.REGULARIZER_TYPE:
            partial_model_generator = generator_helper.get_generator_over_regularizers(model, self.number_of_layers,
                                                                                       self.nodes_per_layer_list,
                                                                                       self.activation_function_name,
                                                                                       self.kernel_initialization_function_name,
                                                                                       self.bias_initialization_function_name,
                                                                                       network_type)
            return self.build_rnn_network_generator_helper(partial_model_generator)

        else:
            raise customexception.NeuralConfigError("Iterating variable not supported or ill-configured")

    def build_dnn_network(self, iterating_variable):
        model = layerutils.get_initialized_model_with_input_layer(self.input_units, self.input_shape)
        if iterating_variable is None:
            final_model = self.build_dnn_network_helper(model)
            final_model.compile(optimizer=self.optimizer_name,
                                loss=self.loss_name,
                                metrics=self.output_metric_name_list)
            return final_model

        elif iterating_variable.lower() == model_constants.OPTIMIZER_TYPE:
            final_model = self.build_dnn_network_helper(model)
            return generator_helper.get_compiled_model_generator_with_optimizer(final_model, self.loss_name, self.output_metric_name_list)

        elif iterating_variable.lower() == model_constants.LOSS_TYPE:
            final_model = self.build_dnn_network_helper(model)
            return generator_helper.get_compiled_model_generator_with_loss_function(final_model, self.optimizer_name,
                                                                                    self.output_metric_name_list, self.output_loss_type)
        elif iterating_variable.lower() == model_constants.INITIALIZER_TYPE:
            partial_model_generator = generator_helper.get_generator_over_initializer(model,self.number_of_layers,self.nodes_per_layer_list,
                                                                                      self.activation_function_name, self.activity_regularizer_name,
                                                                                      model_constants.DNN_MODEL_TYPE)
            return self.build_dnn_network_generator_helper(partial_model_generator)

        elif iterating_variable.lower() == model_constants.ACTIVATION_TYPE:
            partial_model_generator = generator_helper.get_generator_over_activations(model, self.number_of_layers, self.nodes_per_layer_list,
                                                                                      kernel_initializer=self.kernel_initialization_function_name,
                                                                                      bias_initializer=self.bias_initialization_function_name,
                                                                                      activity_regularizer=self.activity_regularizer_name,
                                                                                      model_type=model_constants.DNN_MODEL_TYPE)
            return self.build_dnn_network_generator_helper(partial_model_generator)

        elif iterating_variable.lower() == model_constants.REGULARIZER_TYPE:
            partial_model_generator = generator_helper.get_generator_over_regularizers(model, self.number_of_layers, self.nodes_per_layer_list,
                                                                                       activation_function=self.activation_function_name,
                                                                                       kernel_initializer=self.kernel_initialization_function_name,
                                                                                       bias_initializer=self.bias_initialization_function_name,
                                                                                       model_type=model_constants.DNN_MODEL_TYPE)
            return self.build_dnn_network_generator_helper(partial_model_generator)

        else:
            raise customexception.NeuralConfigError("Iterating Variable not supported or ill-defined")

    def build_dnn_network_helper(self, model):
        partial_built_model = generator_helper.get_dnn_block(model, self.number_of_layers,
                                                             self.nodes_per_layer_list,
                                                             self.activation_function_name,
                                                             self.kernel_initialization_function_name,
                                                             self.bias_initialization_function_name,
                                                             self.activity_regularizer_name)
        final_model = layerutils.add_output_layer(partial_built_model, self.output_units,
                                                  default_activation=self.output_layer_activation,
                                                  kernel_initializer=self.kernel_initialization_function_name,
                                                  bias_initializer=self.bias_initialization_function_name,
                                                  activity_regularizer=self.activity_regularizer_name)
        return final_model

    def build_rnn_network_helper(self, model):
        partial_built_model = generator_helper.get_rnn_block(model, self.number_of_layers,
                                                             self.nodes_per_layer_list,
                                                             self.rnn_cell_type,
                                                             kernel_initializer=self.kernel_initialization_function_name,
                                                             bias_initializer=self.bias_initialization_function_name,
                                                             activity_regularizer=self.activity_regularizer_name)
        final_model = layerutils.add_output_layer(partial_built_model, self.output_units,
                                                  default_activation=self.output_layer_activation,
                                                  kernel_initializer=self.kernel_initialization_function_name,
                                                  bias_initializer=self.bias_initialization_function_name,
                                                  activity_regularizer=self.activity_regularizer_name)
        return final_model

    def build_dnn_network_generator_helper(self, partial_model_generator):
        for partial_model in partial_model_generator:
            final_model = layerutils.add_output_layer(partial_model, self.output_units,
                                                      default_activation=self.output_layer_activation,
                                                      kernel_initializer=self.kernel_initialization_function_name,
                                                      bias_initializer=self.bias_initialization_function_name,
                                                      activity_regularizer=self.activity_regularizer_name)
            final_model.compile(optimizer=self.optimizer_name,
                                loss=self.loss_name,
                                metrics=self.output_metric_name_list)
            yield final_model

    def build_rnn_network_generator_helper(self, partial_model_generator):
        for partial_model in partial_model_generator:
            final_model = layerutils.add_output_layer(partial_model, self.output_units,
                                                      default_activation=self.output_layer_activation,
                                                      kernel_initializer=self.kernel_initialization_function_name,
                                                      bias_initializer=self.bias_initialization_function_name,
                                                      activity_regularizer=self.activity_regularizer_name)
            final_model.compile(optimizer=self.optimizer_name,
                                loss=self.loss_name,
                                metrics=self.output_metric_name_list)
            yield final_model
