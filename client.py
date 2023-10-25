
from typing import Dict, Tuple
import flwr as fl
from flwr.common import Config, NDArrays, Scalar

import torch

from collections import OrderedDict

from model import Net, train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 cid: str,
                 trainloader,
                 valloader,
                 num_classes) -> None:
        super().__init__()

        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader

        self.model = Net(num_classes)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def set_parameters(self, parameters):

        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)


    def get_parameters(self, config: Dict[str, Scalar]):

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    

    def fit(self, parameters, config):

        # copy parameter sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config['lr']
        momentum = config['momentum']
        epochs = config['local_epochs']

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        print("Client CID: ", self.cid, " - trainset length: ", len(self.trainloader.dataset))
        train(self.model, self.trainloader, optimizer, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}
    

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {'accuracy': accuracy}
    
    

def generate_client_fn(trainloaders, valloaders, num_classes):

    def client_fn(cid: str):
        return FlowerClient(cid=cid,
                            trainloader=trainloaders[int(cid)],
                            valloader=valloaders[int(cid)],
                            num_classes=num_classes)

    return client_fn
