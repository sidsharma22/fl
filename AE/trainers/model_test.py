import matplotlib.pyplot as plt
from .model_base import BaseTrainer
import torch


class Test(BaseTrainer):
    def __init__(self) -> None:
        super().__init__()
        
    def test_nn(self, trainloaders, valloaders, testloader, Net, DEVICE):
        trainloader = trainloaders[1]
        valloader = valloaders[1]
        net = Net().to(DEVICE)
        if net.__class__.__name__ == 'LinearM':
            for epoch in range(5):
                self.train(net, trainloader, epochs=5,copt_eps=1e-5,copt_lr=10 ,csa=1)
                loss, accuracy = self.test_linear(net, valloader)
                print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

            loss, accuracy = self.test_linear(net, testloader)
            print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
            
        elif net.__class__.__name__ == 'AE':
            for epoch in range(5):
                self.train(net, trainloader, epochs=10,copt_eps=0,copt_lr=10,csa=0)
                loss, accuracy = self.test_ae(net, valloader)
                print(f"AE Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

            loss, accuracy = self.test_ae(net, testloader)
            print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")