"""xgboost_quickstart: A Flower / XGBoost app."""
import sys

module_path='xgboost_quickstart'

if module_path not in sys.path: 
    sys.path.append(module_path)

import data
from data import *

from logging import INFO

import xgboost as xgb
from flwr.common import log
#from flwr_datasets import FederatedDataset
#from flwr_datasets.partitioner import IidPartitioner

from sklearn.model_selection import train_test_split




def transform_dataset_to_dmatrix(x,y):
    """Transform dataset to DMatrix format for xgboost."""
   
    new_data = xgb.DMatrix(x, label=y)
    return new_data


fds = None  # Cache FederatedDataset


# Fonction pour charger vos propres données par partition
def load_data(partition_id):
    """Load partitioned custom data."""
    
    # Charger les données (par exemple depuis un CSV)
    dt = data.my_data(partition_id)

    X_train, y_train, X_test,y_test = dt.access(partition_id)

    #transfomatation des données en dmatrix
    train_dmatrix = transform_dataset_to_dmatrix(X_train,y_train)
    test_dmatrix = transform_dataset_to_dmatrix(X_test,y_test)
    num_train= X_train.shape[0]
    num_test = X_test.shape[0]

    return train_dmatrix, test_dmatrix, num_train, num_test



def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
