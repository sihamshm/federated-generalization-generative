"""xgboost_quickstart: A Flower / XGBoost app."""

from typing import Dict

from flwr.common import Context, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedXgbBagging

import matplotlib
matplotlib.use('Agg')  # Utiliser un backend sans Tkinter
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

metrics_df = pd.DataFrame()



def evaluate_metrics_aggregation(eval_metrics):
    global metrics_df
    # Transformer les données en un dictionnaire
   
    if metrics_df.empty :
         columns = {}
         for idx, (key, metrics) in enumerate(eval_metrics):
            columns[f'AU-PRC_{idx}'] = metrics['AU-PRC']
            columns[f'AUC_{idx}'] = metrics['AUC']
         metrics_df =  pd.DataFrame(columns, index=[0]) 
        
    else:
         columns = {}
         for idx, (key, metrics) in enumerate(eval_metrics):
            columns[f'AU-PRC_{idx}'] = metrics['AU-PRC']
            columns[f'AUC_{idx}'] = metrics['AUC']
         data=  pd.DataFrame(columns,index=[0])
         metrics_df = pd.concat([metrics_df,data],ignore_index=True)

    print(metrics_df)
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )

    prc_aggregated = (
        sum([metrics["AU-PRC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated,"AU-PRC": prc_aggregated}


    rounds = metrics_df.shape[0]
    if (rounds==50):
        print(metrics_df)
        auroc_plot(metrics_df)

    return metrics_aggregated


def config_func(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config

def auroc_plot(metrics_df):

    key= metrics_df.iloc[:,0]
    plt.figure(figsize=(8,6))
    output_path = os.path.join(os.environ['TMPDIR'], 'data/output/')

    rounds = [a for a in range(1,metrics_df.shape[0]+1)]

   
    plt.plot(rounds, metrics_df['AUC_0'] ,label=f'AU-ROC CAD I25', color='blue')
    plt.plot(rounds, metrics_df['AUC_1'] ,label=f'AU-ROC CKD N18', color='green', linestyle='--', marker='.')

    plt.plot(rounds, metrics_df['AU-PRC_0'], label=f'AU-PRC CAD I25', color='red')  # Ajout des points pour client 2
    plt.plot(rounds, metrics_df['AU-PRC_1'], label=f'AU-PRC CKD N18', color='orange', linestyle='--',marker='.')  # Ajout des points pour client 2
 
    values = [0.76] * len(rounds)  # Une liste de 0.20 ayant la même longueur que rounds
    plt.plot(rounds, values ,label=f'AU-ROC train_CAD Test_CKD ', color='lightblue')
    values = [0.67] * len(rounds)  # Une liste de 0.20 ayant la même longueur que rounds
    plt.plot(rounds, values ,label=f'AU-ROC train_CKD Test_CAD', color='lightgreen', linestyle='--', marker='.')

    values = [0.70] * len(rounds)  # Une liste de 0.20 ayant la même longueur que rounds
    plt.plot(rounds, values , label=f'AU-PRC train_CAD Test_CKD', color='lightsalmon')  # Ajout des points pour client 2
    values = [0.65] * len(rounds)  # Une liste de 0.20 ayant la même longueur que rounds
    plt.plot(rounds, values, label=f'AU-PRC train_CKD Test_CAD', color='#FFDAB9', linestyle='--',marker='.')  # Ajout des points pour client 2


    plt.ylabel("AU-ROC & AU-PRC ")
    plt.xlabel("nombre d'itérations")
    plt.title("AUC-ROC et AUC-PRC en fonction du nombre d'itération ")
    # Afficher la légende à l'extérieur du tracé
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Ajustez les valeurs pour personnaliser la position
    plt.tight_layout(rect=[0, 0, 1, 1])    # Ajuster les marges pour éviter la superposition
    plt.savefig(output_path+"auroc_prc_plot.png")
    print("Le tracé a été bien enregitrer")



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Init an empty Parameter
    parameters = Parameters(tensor_type="", tensors=[])

    # Define strategy
    strategy = FedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(
    server_fn=server_fn,
)
