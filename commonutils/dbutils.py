import hashlib
from constants import dbconstants
from db import mongobase


def dictionary_from_network_builder(network_builder_object):
    database_object_dict = dict({})
    database_object_dict['network_type'] = network_builder_object.network_type
    database_object_dict['number_of_layers'] = network_builder_object.number_of_layers
    database_object_dict['activation_function_name'] = network_builder_object.activation_function_name
    database_object_dict['kernel_initializer'] = network_builder_object.kernel_initializer
    database_object_dict['bias_initializer'] = network_builder_object.bias_initializer
    database_object_dict['activity_regularizer_name'] = network_builder_object.activity_regularizer_name
    database_object_dict['optimizer_name'] = network_builder_object.optimizer_name
    database_object_dict['loss_name'] = network_builder_object.loss_name
    database_object_dict['type_output_loss'] = network_builder_object.type_output_loss
    return database_object_dict


def get_network_hash(network_object_dict):
    network_config_string = ''
    for network_value in network_object_dict.values():
        network_config_string = network_config_string + str(network_value) + '_'
    network_config_string = network_config_string[:-1]
    hash_digest = hashlib.sha1(network_config_string.encode())
    return hash_digest


def get_mongo_connection(db_name=dbconstants.DB_NAME, collection_name=dbconstants.COLLECTION_NAME):
    mongo = mongobase.MongoConnector(dbconstants.LOCAL_MONGO_HOSTNAME, dbconstants.LOCAL_MONGO_PORT)
    mongo.set_db(db_name)
    mongo.set_collection(collection_name)
    return mongo


def check_for_duplicate_and_insert(network_dict):
    query_dict = dict({'network_hash': network_dict['network_hash']})
    mongo = get_mongo_connection()
    if mongo.check_document(query_dict) is False:
        mongo.insert_document(network_dict)
    mongo.close_connection()
