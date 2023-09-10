import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config_fn, get_evaluate_fn

@hydra.main(config_path='conf', config_name='base', version_base=None)

def main(cfg: DictConfig):

    ## 1. Parse config and print experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, 
                                                                  cfg.batch_size)
    
    ## 3. Define clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config_fn(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader)
    )
    
    ## 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy
    )

if __name__ == '__main__':
    main()