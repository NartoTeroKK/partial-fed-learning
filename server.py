
from numpy import exp
from omegaconf import DictConfig
import torch
from collections import OrderedDict
import flwr as fl
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateRes,
    FitRes,
    Scalar
)
from flwr.server.client_proxy import ClientProxy


from model import Net, test

def get_on_fit_config_fn(config: DictConfig):

    def fit_config_fn(server_round: int):
        '''
        k = 0.1
        if server_round==1:
            lr = config.lr
        else:
            lr = config.lr * exp(-k * server_round)
        print("Server round: ", server_round, " - lr: ", lr)
        '''
        
        return {'lr': config.lr, 'momentum': config.momentum,
                'local_epochs': config.local_epochs}
    
    return fit_config_fn

def get_on_evaluate_config_fn(config: DictConfig):

    def evaluate_config_fn(server_round: int):

        return {}

    return evaluate_config_fn


def get_evaluate_fn(num_classes: int, testloader):

    def evaluate_fn(server_round: int, parameters, config):

        model = Net(num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, testloader, device)

        return loss, {'accuracy': accuracy}

    return evaluate_fn

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        aggregated_accuracy = sum(accuracies) / sum(examples)
        print(f"Round {server_round} accuracy aggregated from client results: {aggregated_accuracy}")

        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}