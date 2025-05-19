"""FederatedLearningLR: A Flower / sklearn app."""

import sys

module_path='federatedlearninglr'

if module_path not in sys.path: 
    sys.path.append(module_path)


import numpy as np
from sklearn.linear_model import LogisticRegression


import pandas as pd
from sklearn.model_selection import train_test_split
import data
from data import *

fds = None  # Cache FederatedDataset



# Fonction pour charger vos propres données par partition
def load_data(partition_id):
    """Load partitioned custom data."""
    
    # Charger les données (par exemple depuis un CSV)
    dt = data.my_data(partition_id)
    
    # Diviser les données en 80% entraînement et 20% test
    X_train, y_train, X_test,y_test = dt.access(partition_id)

    return X_train, X_test, y_train, y_test



def get_model(penalty: str, local_epochs: int):

    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
    )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    n_classes = 2  # MNIST has 10 classes
    n_features = 62523  # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
