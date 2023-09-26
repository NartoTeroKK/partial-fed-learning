import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pickle
from pathlib import Path
import flwr as fl
import torch

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config_fn, get_evaluate_fn, AggregateCustomMetricStrategy

@hydra.main(config_path='conf', config_name='base', version_base=None)

def main(cfg: DictConfig):

    ## 1. Parse config and print experiment output dir
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} using PyTorch {torch.__version__} and Flower {fl.__version__}")

    ## 2. Prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, 
                                                                  cfg.batch_size)
    
    ## 3. Define clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Define strategy
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.01,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.02,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config_fn(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )
    
    ## 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': cfg.num_cpus, 'num_gpus': cfg.num_gpus},  
    )

    ## 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'result.pkl'

    results = {'history':history}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()