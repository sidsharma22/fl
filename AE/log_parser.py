import matplotlib.pyplot as plt
import re
import  numpy as np
##################################################
# Automatic log parser and hyperparameter tuner for Flower
# Sidharth Sharma, 2023
##################################################
min_normal_validation_loss = []
min_normal_training_loss = []
min_normal_validation_acc = []
min_normal_training_acc = []

min_sa_validation_loss = []
min_sa_training_loss = []
min_sa_validation_acc = []
min_sa_training_acc = []

min_csa_validation_loss = []
min_csa_training_loss = []
min_csa_validation_acc = []
min_csa_training_acc = []

min_csa_du_validation_loss = []
min_csa_du_training_loss = []
min_csa_du_validation_acc = []
min_csa_du_training_acc = []

def plot_graph(normal, sa,csa,csa_du,metric='Mean_Squared_Error',title='Validation_Loss'):
    x_values = list(range(50))
    plt.clf()
    plt.plot(x_values, normal, label='Normal (ηl=40)')
    plt.plot(x_values, sa, label='SA (ηl=40, η=10^(-3/2))') 
    plt.plot(x_values, csa, label='CSA_SA (ηl=0.1, η=10^(-3/2))')
    plt.plot(x_values, csa_du, label='CSA_DU_SA (ηl=0.1, η=0.1)')
    plt.xlabel('Rounds')
    plt.ylabel(f'{metric}')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'../output/simulation_results/{title}_{metric}.png', dpi=300, format='png', bbox_inches='tight')


def analyze_normal():
    with open('log_[Normal][AE][LDA].txt', 'r') as file:
        # Read the entire contents of the file into a string
        t = []
        global min_normal_validation_loss
        global min_normal_training_loss
        global min_normal_validation_acc
        global min_normal_training_acc
        
        normal = file.read()
        ## Get the hyperparameters
        hp_client = re.findall(r'LR:\s([0-9]\.*[0-9]*)', normal)
        ## Get validation loss: 
        normal_validation_loss = re.findall(r'losses_distributed\s(.*])', normal)
        # ,\s([0-9].[0-9]*e?-?[0-9]*)
        #print(re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', normal_validation_loss[0]))
        m = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', normal_validation_loss[0])
        min_normal_validation_loss_val = float(m[49])
        min_normal_validation_loss = m
        best_i_validation_loss = 0
        for i in range(1,len(normal_validation_loss)):
            m = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', normal_validation_loss[i])
            if min_normal_validation_loss_val > float(m[49]):
                min_normal_validation_loss_val = float(m[49])
                min_normal_validation_loss = m
                best_i_validation_loss = i
        print("Best Val loss: ", min_normal_validation_loss_val)
        print("Best Val Client LR:", hp_client[best_i_validation_loss])
        
        
        ## Get validation acc: 
        # normal_validation_acc = re.findall(r'metrics_distributed\s(.*)', normal)
        '''
        Let's skip acc parsing for Autoencoder
        '''
        ## Get training loss: 
        normal_training_loss = re.findall(r"metrics_distributed_fit\s.*tr_loss':\s(.*),\s't", normal)
        n = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', normal_training_loss[0])
        min_normal_training_loss_val = float(n[49])
        min_normal_training_loss = n
        best_i_training_loss = 0
        for i in range(1,len(normal_validation_loss)):
            n = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', normal_training_loss[i])
            if min_normal_training_loss_val > float(n[49]):
                min_normal_training_loss_val = float(n[49])
                min_normal_training_loss = n
                best_i_training_loss = i
        print("Best Train loss: ", min_normal_training_loss_val)
        print("Best Train Client LR:", hp_client[best_i_training_loss])
        
        min_normal_validation_loss = [float(x) for x in min_normal_validation_loss]
        min_normal_training_loss  = [float(x) for x in min_normal_training_loss]

        ## Get training acc: 
        #normal_training_acc = re.findall(r"metrics_distributed_fit.*tr_accuracy':(.*)}.*", normal)
        '''
        Let's skip acc parsing for Autoencoder
        '''
        ## Get l2_norms: 
        #normal_l2_norms = re.findall(r"metrics_distributed_fit\s.*l2_norms':\s(.*),\s't", normal)

def analyze_sa():
    with open('log_[SA][AE].txt', 'r') as file:
        # Read the entire contents of the file into a string
        t = []
        global min_sa_validation_loss
        global min_sa_training_loss
        global min_sa_validation_acc
        global min_sa_training_acc
        
        sa = file.read()
        ## Get the hyperparameters
        hp_client = re.findall(r'Client\sLR:\s([0-9]\.*[0-9]*)', sa)
        hp_server = re.findall(r'Server\sLR:\s([0-9]\.*[0-9]*)',sa)
        hp_comb = [(x,y) for x,y in zip(hp_client,hp_server)]
        
        ## Get validation loss: 
        sa_validation_loss = re.findall(r'losses_distributed\s(.*])', sa)
        # ,\s([0-9].[0-9]*e?-?[0-9]*)
        #print(re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', normal_validation_loss[0]))
        m = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', sa_validation_loss[0])
        min_sa_validation_loss_val = float(m[49])
        min_sa_validation_loss = m
        best_i_validation_loss = 0
        for i in range(1,len(sa_validation_loss)):
            #print(sa_validation_loss[i])
            m = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', sa_validation_loss[i])
            if len(m) < 50:
                continue
            if min_sa_validation_loss_val > float(m[49]):
                min_sa_validation_loss_val = float(m[49])
                min_sa_validation_loss = m
                best_i_validation_loss = i
        print("[SA]Best Val loss: ", min_sa_validation_loss_val)
        print("[SA]Best Val Client/Server LR:", hp_comb[best_i_validation_loss-1])
        
        
        ## Get validation acc: 
        # normal_validation_acc = re.findall(r'metrics_distributed\s(.*)', normal)
        '''
        #Let's skip acc parsing for Autoencoder
        '''
        ## Get training loss: 
        sa_training_loss = re.findall(r"metrics_distributed_fit\s.*tr_loss':\s(.*),\s't", sa)
        n = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', sa_training_loss[0])
        min_sa_training_loss_val = float(n[49])
        min_sa_training_loss = n
        best_i_training_loss = 0
        for i in range(1,len(sa_training_loss)):
            n = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', sa_training_loss[i])
            if len(n) < 50:
                continue
            if min_sa_training_loss_val > float(n[49]):
                min_sa_training_loss_val = float(n[49])
                min_sa_training_loss = n
                best_i_training_loss = i
        print("[SA]Best Train loss: ", min_sa_training_loss_val)
        print("[SA]Best Train Client/Server LR:", hp_comb[best_i_training_loss-1])
        
        min_sa_validation_loss = [float(x) for x in min_sa_validation_loss]
        min_sa_training_loss  = [float(x) for x in min_sa_training_loss]

        ## Get training acc: 
        #normal_training_acc = re.findall(r"metrics_distributed_fit.*tr_accuracy':(.*)}.*", normal)
        '''
        #Let's skip acc parsing for Autoencoder
        '''
        ## Get l2_norms: 
        #normal_l2_norms = re.findall(r"metrics_distributed_fit\s.*l2_norms':\s(.*),\s't", normal)
        
def analyze_csa():
   with open('log_[CSA][AE].txt', 'r') as file:
        # Read the entire contents of the file into a string
        t = []
        global min_csa_validation_loss
        global min_csa_training_loss
        global min_csa_validation_acc
        global min_csa_training_acc
        
        csa = file.read()
        ## Get the hyperparameters
        hp_client = re.findall(r'Client\sLR:\s([0-9]\.*[0-9]*)', csa)
        hp_server = re.findall(r'Server\sLR:\s([0-9]\.*[0-9]*)',csa)
        hp_comb = [(x,y) for x,y in zip(hp_client,hp_server)]
        ## Get validation loss: 
        csa_validation_loss = re.findall(r'losses_distributed\s(.*])', csa)
        # ,\s([0-9].[0-9]*e?-?[0-9]*)
        #print(re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', normal_validation_loss[0]))
        m = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', csa_validation_loss[0])
        min_csa_validation_loss_val = float(m[49])
        min_csa_validation_loss = m
        best_i_validation_loss = 0
        #print(min_csa_validation_loss_val)
        for i in range(1,len(csa_validation_loss)):
            #print(sa_validation_loss[i])
            m = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', csa_validation_loss[i])
            if len(m) < 50:
                continue
            #print("Current Value:",min_csa_validation_loss_val)
            #print("Comp value:" ,float(m[49]))
            if min_csa_validation_loss_val > float(m[49]):
                min_csa_validation_loss_val = float(m[49])
                min_csa_validation_loss = m
                best_i_validation_loss = i
        print("[CSA]Best Val loss: ", min_csa_validation_loss_val)
        print("[CSA]Best Val Client/Server LR:", hp_comb[best_i_validation_loss-1])
        
        
        ## Get validation acc: 
        # normal_validation_acc = re.findall(r'metrics_distributed\s(.*)', normal)
        '''
        #Let's skip acc parsing for Autoencoder
        '''
        ## Get training loss: 
        csa_training_loss = re.findall(r"metrics_distributed_fit\s.*tr_loss':\s(.*),\s't", csa)
        n = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', csa_training_loss[0])
        min_csa_training_loss_val = float(n[49])
        min_csa_training_loss = n
        best_i_training_loss = 0
        for i in range(1,len(csa_training_loss)):
            n = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', csa_training_loss[i])
            if len(n) < 50:
                continue
            if min_csa_training_loss_val > float(n[49]):
                min_csa_training_loss_val = float(n[49])
                min_csa_training_loss = n
                best_i_training_loss = i
        print("[CSA]Best Train loss: ", min_csa_training_loss_val)
        print("[CSA]Best Train Client/Server LR:", hp_comb[best_i_training_loss-1])
        min_csa_validation_loss = [float(x) for x in min_csa_validation_loss]
        min_csa_training_loss  = [float(x) for x in min_csa_training_loss]
        ## Get training acc: 
        #normal_training_acc = re.findall(r"metrics_distributed_fit.*tr_accuracy':(.*)}.*", normal)
        '''
        #Let's skip acc parsing for Autoencoder
        '''
        ## Get l2_norms: 
        #normal_l2_norms = re.findall(r"metrics_distributed_fit\s.*l2_norms':\s(.*),\s't", normal)
        

def analyze_csa_du():
   with open('log_[CSA-DU][AE].txt', 'r') as file:
        # Read the entire contents of the file into a string
        t = []
        global min_csa_du_validation_loss
        global min_csa_du_training_loss
        global min_csa_du_validation_acc
        global min_csa_du_training_acc
        
        csa_du = file.read()
        ## Get the hyperparameters
        hp_client = re.findall(r'Client\sLR:\s([0-9]\.*[0-9]*)', csa_du)
        hp_server = re.findall(r'Server\sLR:\s([0-9]\.*[0-9]*)',csa_du)
        hp_comb = [(x,y) for x,y in zip(hp_client,hp_server)]
        
        ## Get validation loss: 
        csa_du_validation_loss = re.findall(r'losses_distributed\s(.*])', csa_du)
        # ,\s([0-9].[0-9]*e?-?[0-9]*)
        #print(re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', normal_validation_loss[0]))
        m = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', csa_du_validation_loss[0])
        min_csa_du_validation_loss_val = float(m[49])
        min_csa_du_validation_loss = m
        best_i_validation_loss = 0
        for i in range(1,len(csa_du_validation_loss)):
            #print(sa_validation_loss[i])
            m = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', csa_du_validation_loss[i])
            if len(m) < 50:
                continue
            if min_csa_du_validation_loss_val > float(m[49]):
                min_csa_du_validation_loss_val = float(m[49])
                min_csa_du_validation_loss = m
                best_i_validation_loss = i
        print("[CSA-DU]Best Val loss: ", min_csa_du_validation_loss_val)
        print("[CSA-DU]Best Val Client/Server LR:", hp_comb[best_i_validation_loss-1])
        
        
        ## Get validation acc: 
        # normal_validation_acc = re.findall(r'metrics_distributed\s(.*)', normal)
        '''
        #Let's skip acc parsing for Autoencoder
        '''
        ## Get training loss: 
        csa_du_training_loss = re.findall(r"metrics_distributed_fit\s.*tr_loss':\s(.*),\s't", csa_du)
        n = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', csa_du_training_loss[0])
        min_csa_du_training_loss_val = float(n[49])
        min_csa_du_training_loss = n
        best_i_training_loss = 0
        for i in range(1,len(csa_du_training_loss)):
            n = re.findall(r',\s([0-9].[0-9]*e?-?[0-9]*)', csa_du_training_loss[i])
            if len(n) < 50:
                continue
            if min_csa_du_training_loss_val > float(n[49]):
                min_csa_du_training_loss_val = float(n[49])
                min_csa_du_training_loss = n
                best_i_training_loss = i
        print("[CSA-DU]Best Train loss: ", min_csa_du_training_loss_val)
        print("[CSA-DU]Best Train Client/Server LR:", hp_comb[best_i_training_loss-1])
        
        min_csa_du_validation_loss = [float(x) for x in min_csa_du_validation_loss]
        min_csa_du_training_loss  = [float(x) for x in min_csa_du_training_loss]

        ## Get training acc: 
        #normal_training_acc = re.findall(r"metrics_distributed_fit.*tr_accuracy':(.*)}.*", normal)
        '''
        #Let's skip acc parsing for Autoencoder
        '''
        ## Get l2_norms: 
        #normal_l2_norms = re.findall(r"metrics_distributed_fit\s.*l2_norms':\s(.*),\s't", normal)


def main():
    
    analyze_normal()
    analyze_sa()
    analyze_csa()
    #analyze_csa_du()
    
    # Plot validation loss
    plot_graph(min_normal_validation_loss,min_sa_validation_loss,min_csa_validation_loss,np.zeros((50), dtype=np.int32),'Mean_Squared_Error','Validation_Loss')
    # Plot training loss
    plot_graph(min_normal_training_loss,min_sa_validation_loss,min_csa_validation_loss,np.zeros((50), dtype=np.int32),'Mean_Squared_Error','Training_Loss')    
    # Plot validation acc
    #plot_graph(min_normal_validation_acc,np.zeros((50), dtype=np.int32),np.zeros((50), dtype=np.int32),np.zeros((50), dtype=np.int32),'Mean_Squared_Error','Validation_Loss')
    # Plot training acc
    #plot_graph(min_normal_training_acc,np.zeros((50), dtype=np.int32), np.zeros((50), dtype=np.int32),np.zeros((50), dtype=np.int32),'Mean_Squared_Error','Training_Loss')  



if __name__ == "__main__":
    main()