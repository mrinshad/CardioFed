import flwr as fl
from flwr.server.strategy import FedXgbBagging
import matplotlib.pyplot as plt
import numpy as np
import time


# FL experimental settings
pool_size = 4
num_rounds = 20
num_clients_per_round = 4
num_evaluate_clients = 4

x=np.arange(1,num_rounds+1)
y=[]
time_taken=[]

def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    y.append(auc_aggregated)
    print("\nAccuracy Score : ",auc_aggregated)
    print("\n")
    return metrics_aggregated


# Define strategy
strategy = FedXgbBagging(
    fraction_fit=(float(num_clients_per_round) / pool_size),
    min_fit_clients=num_clients_per_round,
    min_available_clients=pool_size,
    min_evaluate_clients=num_evaluate_clients,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
)
start_time = time.time()
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8085",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
time_taken.append(time.time() - start_time)
print("Time taken:", time_taken[0], "seconds")
plt.plot(x,y,color='red')
plt.grid(color="green",linewidth=0.5,linestyle='--')
plt.xlabel('number of epoch')
plt.ylabel('Accuracy score')
plt.title('Federated Learning')

plt.show()