from collections import OrderedDict
from typing import List, Tuple
from flwr.common.logger import log
from logging import WARNING

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
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

class LinearM(nn.Module):
    def __init__(self) -> None:
        super(LinearM, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(10000, 500)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return  logits

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        
        self.autoencoder = nn.Sequential(
          nn.Linear(784, 1000),
          nn.LeakyReLU(),
          nn.Linear(1000, 500),
          nn.LeakyReLU(),
          nn.Linear(500, 250),
          nn.LeakyReLU(),
          nn.Linear(250, 30),
          nn.Linear(30, 250),
          nn.LeakyReLU(),
          nn.Linear(250, 500),
          nn.LeakyReLU(),
          nn.Linear(500, 1000),
          nn.LeakyReLU(),
          nn.Linear(1000, 784),
          nn.Sigmoid()
        )
        #for layer in self.autoencoder:
        #    if isinstance(layer,nn.Linear):
        #        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))

        '''
        self.autoencoder = nn.Sequential(
          nn.Linear(784, 32),
          nn.LeakyReLU(),
          nn.Linear(32, 784),
          nn.Sigmoid()
        )
        '''
    def forward(self, x):
        x = x.view(-1, 784)  # (batch size, 784)
        output = self.autoencoder(x)
        return output  # (batch size, 784)

class BaseTrainer(): 
    def train(self,net, trainloader, epochs: int, copt_eps,copt_lr,csa,verbose=False):
        """Train the network on the training set."""
        #log(WARNING,f"Client Learning Rate: {copt_lr}")
        if net.__class__.__name__ == 'AE':
            criterion = torch.nn.MSELoss() # CrossEntropyLoss()
        elif net.__class__.__name__ == 'LinearM':
            criterion = torch.nn.CrossEntropyLoss()
        
        if csa == 0 or csa == -1:
            optimizer = torch.optim.SGD(net.parameters(),lr=copt_lr) ### [(-)0.009, (-)0.005, 0.01, 0.05, 0.1, 0.3, 0.5&&,(-)0.9]
        elif csa == 1:
            optimizer = torch.optim.Adagrad(net.parameters(), lr=copt_lr,eps=copt_eps) ## [0.001, 0.005, 0.009^, 0.01, 0.05, 0.09, 0.1, 0.3, 0.25**,0.5]
        else:
            optimizer = AdagradOptimizer(net, lr=copt_lr, epsilon=copt_eps) ## [0.001, 0.005, 0.009^, 0.01, 0.05, 0.09, 0.1, 0.3, 0.25**,0.5]
        
        global_params = [val.detach().clone() for val in net.parameters()]
        if net.__class__.__name__  == 'AE':
            self._train_one_epoch_ae(net, trainloader, criterion, optimizer, global_params,epochs,csa=csa,verbose=False)
        elif net.__class__.__name__ == 'LinearM':
            self._train_one_epoch_linear(net, trainloader, criterion, optimizer, global_params,epochs,csa=csa,verbose=False)


    def _train_one_epoch_ae(self,net, trainloader, criterion, optimizer, global_params,epochs,csa,verbose=False):
        """
        This function is to avoid the backward parameter copying that happens
        when I save the global parameters from the local parameters.
        Therefore in the train_setup function save the global parameters from the local
        parameters and in this function do the actual training.
        """
        net.train()
        tr_loss = 0
        tr_accuracy = 0
        k = 2
        #bcount = 0
        correct, total, epoch_loss = 0, 0, 0.0
        for epoch in range(epochs):
            for inputs, _ in trainloader:
                inputs = inputs.to(DEVICE)
                outputs = net(inputs)
                inputs = inputs.reshape(-1, 784)
                #loss = criterion(outputs, labels)
                loss = criterion(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                if csa == 2:
                    optimizer.step(k,epoch)
                else:
                    optimizer.step() 
                # Metrics
                epoch_loss += loss
                if False:
                    if bcount == 19:
                        for param in net.parameters():
                            print(f"Epoch {epoch}: Params After Preconditioner Update: {param.data[0]}")
                            break
                #bcount +=1
                #total += labels.size(0)
                #correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            #bcount= 0
            epoch_loss /= len(trainloader)
            #print("*** train  loss: ", epoch_loss)
            epoch_acc = 0 #correct / total
            if verbose:
                log(WARNING,f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
            ## Total loss:
            #tr_loss+=epoch_loss
            #tr_accuracy+=0
            #self.reconstruction_output(images,outputs)
            #return tr_loss/epochs, tr_accuracy/epochs


    def _train_one_epoch_linear(self,net, trainloader, criterion, optimizer, global_params,epochs,csa,verbose=False):
        """
        This function is to avoid the backward parameter copying that happens
        when I save the global parameters from the local parameters.
        Therefore in the train_setup function save the global parameters from the local
        parameters and in this function do the actual training.
        """
        #print("linear is called, with csa option:", csa)
        #print(optimizer.__class__.__name__)
        net.train()
        tr_loss = 0
        tr_accuracy = 0
        k = 2
        #bcount = 0
        correct, total, epoch_loss = 0, 0, 0.0
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                inputs = inputs.to(DEVICE)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                if csa == 2:
                    optimizer.step(k,epoch)
                else:
                    optimizer.step() 
                # Metrics
                epoch_loss += loss
                if False:
                    if bcount == 19:
                        for param in net.parameters():
                            print(f"Epoch {epoch}: Params After Preconditioner Update: {param.data[0]}")
                            break
                #bcount +=1
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            #bcount= 0
            epoch_loss /= len(trainloader)
            epoch_acc = correct/total
            if verbose:
                log(WARNING,f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
            ## Total loss:
        #tr_loss+=epoch_loss
        #tr_accuracy+=0
        #self.reconstruction_output(images,outputs)
        #return tr_loss/epochs, tr_accuracy/epochs
    def test(self,net, testloader):
        loss = 0
        acc = 0
        if net.__class__.__name__  == 'AE':
            loss,acc = self.test_ae(net,testloader)
        elif net.__class__.__name__  == 'LinearM':
            loss,acc = self.test_linear(net,testloader) 
        return loss, acc
    

    def test_ae(self,net, testloader):
        """Evaluate the network on the entire test set."""
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.MSELoss() # CrossEntropyLoss()

        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for images, _ in testloader:
                images = images.to(DEVICE)
                outputs = net(images)
                images = images.reshape(-1, 784)
                loss += criterion(outputs, images).item()
                #_, predicted = torch.max(outputs.data, 1)
                #total += labels.size(0)
                #correct += (predicted == labels).sum().item()
        loss /= len(testloader)
        #accuracy = correct / total
        return loss, 0

    def test_linear(self,net, testloader):
        """Evaluate the network on the entire test set."""
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.CrossEntropyLoss()

        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for inputs,labels in testloader:
                inputs = inputs.to(DEVICE)
                outputs = net(inputs)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader)
        accuracy = correct / total
        return loss, accuracy

    def reconstruction_output(self,images, outputs):
        n_images = 15
        image_width = 28
        plt.clf()
        fig, axes = plt.subplots(nrows=2, ncols=n_images, 
                                sharex=True, sharey=True, figsize=(20, 2.5))
        orig_images = images[:n_images]
        decoded_images = outputs[:n_images]

        for i in range(n_images):
            for ax, img in zip(axes, [orig_images, decoded_images]):
                curr_img = img[i].detach().to(torch.device('cpu'))
                ax[i].imshow(curr_img.view((image_width, image_width)), cmap='binary')
        plt.savefig('../output/ae_output/ae_reconstruction_data_samples.png', dpi=300, format='png', bbox_inches='tight')


class AdagradOptimizer:
    count = 0
    def __init__(self, model, lr=0.01, epsilon=0.01,):
        self.model = model
        self.lr = lr
        self.epsilon = epsilon
        self.sum_of_squared_gradients = {}


        for param in model.parameters():
            self.sum_of_squared_gradients[param] = torch.zeros_like(param)

    def step(self,k=2,i=0):
        for param in self.model.parameters():
            if param.grad is None:
                continue
            if i % k == 0:
              self.sum_of_squared_gradients[param] += param.grad.pow(2)
            adjusted_lr = self.lr / (torch.sqrt(self.sum_of_squared_gradients[param]) + self.epsilon)
            param.data -= adjusted_lr * param.grad

    def zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()