

### Non-Flower Setup
##[0] Read the Config file to define the parameters through yml 
##[1] Data
##  [a] Setup Data
##  [b] Visualise data (non-IID) specifically
##[2] Model
##  [a] Setup Model
##  [b] Test Model
##  [c] Visualise the model reconstruction output 

### Flower Setup
##[1] Flower Utilities
##  [a] Set and Get parameters 
##  [b] Define FlowerClient 
##[2] Flower Metrics
##  [a] Fit and Evaluate metric functions
##[3] Run Simulation
##  [a] Define Client Function 
##  [b] Define Strategy
##  [c] Define and Simulation func
import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.dataset import Mnist_data, Stackoverflow
from data.lda import Preprocess
from trainers.model_test import Test
from trainers.model_base import AE, LinearM
from flower.run_simulation import Sim

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("In main")
    preprocess = Preprocess()
    test = Test()
    NUM_CLIENTS: int = cfg.num_clients
    BATCH_SIZE: int  = cfg.batch_size
    DATASET: str     = cfg.dataset
    MODEL: str       = cfg.model
    ADAPTIVITY: str  = cfg.adaptivity
    CLIENT_LR:  list   = cfg.client_lr 
    SERVER_LR:  list   = cfg.server_lr 
    CLIENT_EPS: list  = cfg.client_eps 
    SERVER_EPS: list   = cfg.server_eps 
    DEVICE: str      = cfg.device
    IID: bool   = cfg.iid  ## Create non-IID dataset
    print(type(CLIENT_LR))
    
    ## Data
    #-setup_data()
    if DATASET == 'MNIST':
        data = Mnist_data(NUM_CLIENTS,IID,BATCH_SIZE)
        model = AE
    elif DATASET == 'StackOverflow':
        data = Stackoverflow(NUM_CLIENTS,BATCH_SIZE)
        model = LinearM

    trainloaders, valloaders, testloader = data.load_datasets()
    #-viz_data()
    ## Check Samples
    #preprocess.visualize_data(trainloaders=trainloaders)
    ## Check train distribution
    #preprocess.visualize_trainloader(trainloaders=trainloaders)
    ## Check val distribution
    #preprocess.visualize_valloader(valloaders=valloaders)
    print("*******************************")
    print("*****[Data Setup Done]*****")
    print("*******************************")

    ## Model
    #-setup_model()
    # [TODO] Make sure the correct model is instantiated 
    #-test_model()
    #-viz_model()
    print("*******************************")
    print("*****[Centralized Test on Client 1 Data]*****")
    print("*******************************")
    test.test_nn(trainloaders=trainloaders,valloaders=valloaders,testloader=testloader,Net=model,DEVICE=DEVICE)
    print("*******************************")
    print("*****[Model Setup Done]*****")
    print("*******************************")

    ###
    # Flower
    ###
    #-Check if flower utilities are defined 
    #-run_simulation()
    print("************************************")
    print("*****[Flower Simulation Start]*****")
    print("************************************")
    
    sim = Sim(CLIENT_LR, SERVER_LR, CLIENT_EPS, SERVER_EPS,model)
'''
    ## [CASE 1] Normal
    if  ADAPTIVITY == 'normal':
        sim.run(trainloaders=trainloaders,valloaders=valloaders)
    ## [CASE 2] Server Side Adaptivity
    elif  ADAPTIVITY == 'sa':
        sim.run_sa(trainloaders=trainloaders,valloaders=valloaders,csa=0)
    ## [CASE 3] Client Side Adaptivity with Server Side Adaptivity
    elif  ADAPTIVITY == 'csa':
        sim.run_sa(trainloaders=trainloaders,valloaders=valloaders,csa=1)
    ## [CASE 4] Delayed Updates on Client Side with Server Side Adaptivity
    elif  ADAPTIVITY == 'csa_du':
        sim.run_sa(trainloaders=trainloaders,valloaders=valloaders,csa=2)
    else:
        print("Invalid Adaptivity Option selected")
'''
if __name__ == "__main__":
    main()

