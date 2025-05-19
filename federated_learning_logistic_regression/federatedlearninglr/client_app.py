"""fdlr: A Flower / sklearn app."""

import warnings

from sklearn.metrics import log_loss

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from federatedlearninglr.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)


from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

from sklearn import metrics



class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        y_pred_proba = self.model.predict_proba(self.X_test)  # Probabilités pour log_loss et ROC AUC
        #y_pred = self.model.predict(self.X_test)  # Prédictions binaires pour précision, rappel, etc.

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        #accuracy = self.model.score(self.X_test, self.y_test)
        #precision = precision_score(self.y_test, y_pred)
        #recall = recall_score(self.y_test, y_pred)

            # Calcul de l'AUC et de l'AU-PRC
        auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])  # L'AUC est basé sur les probabilités de la classe positive
        auprc = average_precision_score(self.y_test, y_pred_proba[:, 1])  # AU-PRC basé sur les probabilités de la classe positive
        metrics_site = {

            "AUC": auc,
            "AU-PRC": auprc
        }
        # Retourner les résultats
        return loss, len(self.X_test),metrics_site





def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    print("+++++++++++++++++",partition_id)
    num_partitions = context.node_config["num-partitions"]
    print("--------------",num_partitions)


    X_train, X_test, y_train, y_test = load_data(partition_id)

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
