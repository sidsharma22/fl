import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import copy
import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import NDArrays
from trainers.model_base import BaseTrainer 
from typing import List, Optional, Tuple, Union
from collections import OrderedDict


NUM_CLIENTS = 10
class Flwr_base():
    def __init__(self) -> None:
        pass
    def get_parameters(self,net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self,net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        norms = [m["l2_norm"] for _, m in metrics]
        tr_loss = [num_examples * m["tr_loss"] for num_examples, m in metrics]
        tr_accuracy = [num_examples * m["tr_accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        return {"l2_norms": sum(norms)/NUM_CLIENTS, "tr_loss": sum(tr_loss)/sum(examples), "tr_accuracy":sum(tr_accuracy)/sum(examples)}

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}
        
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader,copt_eps=0.01,copt_lr=0.01,csa=0):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.copt_eps = copt_eps
        self.copt_lr = copt_lr
        self.csa = csa
        self.bt = BaseTrainer()
        self.flwr_base = Flwr_base()
        
    def get_parameters(self, config):
        return self.flwr_base.get_parameters(self.net)

    def fit(self, parameters, config):
        l2_norm = 0
        original_params = copy.deepcopy(parameters)
        self.flwr_base.set_parameters(self.net, parameters)
        tr_loss, tr_accuracy = self.bt.test(self.net, self.trainloader)
        self.bt.train(self.net, self.trainloader, epochs=2,copt_eps=self.copt_eps,copt_lr=self.copt_lr,csa=self.csa)
        updated_params = self.get_parameters(self.net)
        update = [np.subtract(x, y) for (x, y) in zip(updated_params, original_params)]
        l2_norm = self._get_update_norm(update)
        if self.csa == -1:
            return updated_params, len(self.trainloader), {"l2_norm": float(l2_norm), "tr_loss": float(tr_loss), "tr_accuracy": float(tr_accuracy)}
        else:
            return update, len(self.trainloader), {"l2_norm": float(l2_norm),"gp":original_params, "tr_loss": float(tr_loss), "tr_accuracy": float(tr_accuracy)}

    def evaluate(self, parameters, config):
        self.flwr_base.set_parameters(self.net, parameters)
        loss, accuracy = self.bt.test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def _get_update_norm(self,update: NDArrays) -> float:
      flattened_update = update[0]
      for i in range(1, len(update)):
          flattened_update = np.append(flattened_update, update[i])  # type: ignore
      return float(np.sqrt(np.sum(np.square(flattened_update))))
  

