import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import copy
import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import NDArrays
from trainers.model_base import BaseTrainer
from flwr.common.logger import log
from logging import WARNING


from .flwr_base import FlowerClient, Flwr_base
from .custom_fedavg import FedAvg
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
#print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

##
# ηl ∈{10^−1.5,10^−1,...,10^2}
# η ∈ {10^−2,10−1,.5,...,10^1}#x
##
NUM_CLIENTS = 10


class Sim():
    def __init__(self,copt_lr_range,sopt_lr_range,copt_eps_range,sopt_tau_range,Net) -> None:
        self.copt_lr_range = copt_lr_range
        self.sopt_lr_range = sopt_lr_range
        self.copt_eps_range = copt_eps_range
        self.sopt_tau_range = sopt_tau_range
        self.Net = Net
        
    
    def run(self,trainloaders,valloaders):
        fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"log_[Normal][AE][LDA].txt")
        for copt_lr in self.copt_lr_range: # ,0.1,0.0316227766,0.01 [0.01,0.0316227766,0.1,1,5,10]:  ###(0.3) best tr accuracy 0.995 and loss 0.000262 [0.2,0.3,0.4,0.5,0.7]
                copt_eps=0
                copt_lr=copt_lr
                log(WARNING,f"*****[CLIENT] Client eps: {copt_eps} and Client LR: {copt_lr}*****")

                def client_fn(cid: str) -> FlowerClient:
                    """Create a Flower client representing a single organization."""
                    net = self.Net().to(DEVICE)

                    # Load data
                    # Note: each client gets a different trainloader/valloader, so each client
                    # will train and evaluate on their own unique data
                    trainloader = trainloaders[int(cid)]
                    valloader = valloaders[int(cid)]
                    # log(WARNING,f"*****Client eps: {copt_eps} and Client LR: {copt_lr}*****")
                    # Create a  single Flower client representing a single organization
                    return FlowerClient(net, trainloader, valloader,copt_eps,copt_lr,csa=-1)

                #log(WARNING,f"*****Server eps: {sopt_tau} and Server LR: {sopt_lr}*****")
                # Create FedAvg strategy
                strategy = fl.server.strategy.FedAvg(
                    fraction_fit=1.0,  # Sample 100% of available clients for training
                    fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
                    min_fit_clients=10,  # Never sample less than 10 clients for training
                    min_evaluate_clients=10,  # Never sample less than 5 clients for evaluation
                    min_available_clients=10,  # Wait until all 10 clients are available
                    evaluate_metrics_aggregation_fn=Flwr_base.weighted_average,
                    fit_metrics_aggregation_fn = Flwr_base.fit_metrics,
                )

                # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
                client_resources = None
                if DEVICE.type == "cuda":
                    client_resources = {"num_gpus":1,
                                        "num_cpus":2}
                if DEVICE.type == "cpu":
                    client_resources = {"num_gpus":0,
                                        "num_cpus":8} 
                if DEVICE.type == "mps":
                    client_resources = {"num_gpus":0,
                                        "num_cpus":8}     
                ## Add something here for mps       
                # Start simulation
                fl.simulation.start_simulation(
                    client_fn=client_fn,
                    num_clients=NUM_CLIENTS,
                    config=fl.server.ServerConfig(num_rounds=10),
                    strategy=strategy,
                    client_resources=client_resources,
                )

    def run_sa(self,trainloaders,valloaders,csa):
        if csa == 0:
            case = '[SA][AE]'
        elif csa == 1:
            case = '[CSA][AE]'
        else:
            case = '[CSA-DU][AE]'
        fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"log_{case}.txt")

        #ηl ∈{0.0316227766,10−1,1,5,10,100} 
        #η ∈ {0.01,0.1,0.5,1,5,10}
        for copt_eps in self.copt_eps_range:
            for sopt_tau in self.sopt_tau_range:
                for sopt_lr in  self.sopt_lr_range:  # this already ran for csa-du so removing them for now.
                    for copt_lr in  self.copt_lr_range:        #[1e-2, 1e-1, 0.5]:
                        copt_eps=copt_eps
                        copt_lr=copt_lr
                        log(WARNING,f"*****[CLIENT] Client eps: {copt_eps} and Client LR: {copt_lr}*****")
                        log(WARNING,f"*****[SERVER] Server eps: {sopt_tau} and Server LR: {sopt_lr}*****")

                        def client_fn(cid: str) -> FlowerClient:
                            """Create a Flower client representing a single organization."""
                            # Load model
                            net = self.sNet().to(DEVICE)

                            # Load data 
                            # Note: each client gets a different trainloader/valloader, so each client
                            # will train and evaluate on their own unique data
                            trainloader = trainloaders[int(cid)]
                            valloader = valloaders[int(cid)]
                            #log(WARNING,f"*****Client eps: {copt_eps} and Client LR: {copt_lr}*****")
                            # Create a  single Flower client representing a single organization
                            return FlowerClient(net, trainloader, valloader,copt_eps,copt_lr,csa)

                        #log(WARNING,f"*****Server eps: {sopt_tau} and Server LR: {sopt_lr}*****")
                        # Create FedAvg strategy
                        strategy = FedAvg(
                            fraction_fit=1.0,  # Sample 100% of available clients for training
                            fraction_evaluate=1.0,  # Sample 50% of available clients for evaluation
                            min_fit_clients=10,  # Never sample less than 10 clients for training
                            min_evaluate_clients=10,  # Never sample less than 5 clients for evaluation
                            min_available_clients=10,  # Wait until all 10 clients are available
                            evaluate_metrics_aggregation_fn=Flwr_base.weighted_average,
                            fit_metrics_aggregation_fn = Flwr_base.fit_metrics,
                            sopt_lr=sopt_lr,
                            sopt_tau=sopt_tau,

                        )

                        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
                        client_resources = None
                        if DEVICE.type == "cuda":
                            client_resources = {"num_gpus": 1}
                        if DEVICE.type == "cpu":
                            client_resources = {"num_gpus":0,
                                                "num_cpus":8}  
                        if DEVICE.type == "mps":
                            client_resources = {"num_gpus":0,
                                                "num_cpus":8} 
                        # Start simulation
                        fl.simulation.start_simulation(
                            client_fn=client_fn,
                            num_clients=NUM_CLIENTS,
                            config=fl.server.ServerConfig(num_rounds=50),
                            strategy=strategy,
                            client_resources=client_resources,
                        )