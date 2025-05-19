"""FederatedLearningLR: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federatedlearninglr.task import get_model, get_model_params, set_initial_params

import matplotlib
matplotlib.use('Agg')  # Utiliser un backend sans Tkinter
import matplotlib.pyplot as plt

import pandas as pd
import sys
import os

metrics_df = pd.DataFrame()

def aggregate_evaluate_metrics(eval_metrics):
    # Exemple : moyenne pondérée basée sur le nombre de données par client
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


    total_examples = sum(num for num,_ in eval_metrics)

    roc = sum(metrics["AUC"] * num for num, metrics in eval_metrics) / total_examples
    prc = sum(metrics["AU-PRC"] * num for num, metrics in eval_metrics) / total_examples

    # Retourner l'accuracy agrégée sous forme de dictionnaire
    metrics_aggregated = {"AUC": roc, "AU-PRC": prc}
    rounds = metrics_df.shape[0]
    if (rounds==50):
        print(metrics_df)
        auroc_plot(metrics_df)

    return metrics_aggregated 

def auroc_plot(metrics_df):
    key= metrics_df.iloc[:,0]
    plt.figure(figsize=(8,6))
    output_path = os.path.join(os.environ['TMPDIR'], 'data/output')

    #plt.plot([0, metrics_df.shape[0]], [0,1],'r--')
    rounds = [a for a in range(1,metrics_df.shape[0]+1)]

    plt.plot(rounds, metrics_df['AUC_0'] ,label=f'AU-ROC CAD I25', color='blue')
    plt.plot(rounds, metrics_df['AUC_1'] ,label=f'AU-ROC COPD J44', color='green', linestyle='--', marker='.')
    plt.plot(rounds, metrics_df['AU-PRC_0'], label=f'AU-PRC CAD I25', color='red')  # Ajout des points pour client 2
    plt.plot(rounds, metrics_df['AU-PRC_1'], label=f'AU-PRC COPD J44', color='orange', linestyle='--', marker='.')  # Ajout des points pour client 2

    plt.ylabel("AU-ROC & AU-PRC")
    plt.xlabel("nombre d'itérations")
    plt.title("AUC-ROC et AUC-PRC en fonction du nombre d'itération ")
    plt.legend()
    plt.savefig(output_path+"auroc_prc_plot.png")
    print("Le tracé a été bien enregitrer")



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
