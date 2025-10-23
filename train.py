import os
import torch
import torch_geometric
import random
import time
import numpy as np

from graph_data_loader import GraphDataset
from GCN import GNNPolicy

torch.cuda.empty_cache()

torch.backends.cudnn.enabled = True  
torch.backends.cudnn.benchmark = True 


train_task=f'train'
if not os.path.isdir(f'./train_logs'):
    os.mkdir(f'./train_logs')
if not os.path.isdir(f'./train_logs/{train_task}'):
    os.mkdir(f'./train_logs/{train_task}')
if not os.path.isdir(f'./pretrain'):
    os.mkdir(f'./pretrain') 
if not os.path.isdir(f'./pretrain/{train_task}'):
    os.mkdir(f'./pretrain/{train_task}')

valid_task=f'test'
if not os.path.isdir(f'./valid_logs'):
    os.mkdir(f'./valid_logs')
if not os.path.isdir(f'./valid_logs/{valid_task}'):
    os.mkdir(f'./valid_logs/{valid_task}')
if not os.path.isdir(f'./pretrain'):
    os.mkdir(f'./pretrain') 
if not os.path.isdir(f'./pretrain/{valid_task}'):
    os.mkdir(f'./pretrain/{valid_task}')

    
model_save_path = f'./pretrain/{train_task}/'
log_save_path = f"train_logs/{train_task}/"
log_file = open(f'{log_save_path}{train_task}_train.log', 'wb')


LEARNING_RATE = 0.001  
NB_EPOCHS = 10000  
BATCH_SIZE = 8       
WEIGHT_NORM = 100    

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DIR_BG_train = f'./dataset/{train_task}/BG'    
DIR_SOL_train = f'./dataset/{train_task}/solution' 
sample_names_train = os.listdir(DIR_BG_train)

DIR_BG_valid = f'./dataset/{valid_task}/BG'   
DIR_SOL_valid = f'./dataset/{valid_task}/solution'
sample_names_valid = os.listdir(DIR_BG_valid)  


sample_files_train = [(os.path.join(DIR_BG_train, name), 
               os.path.join(DIR_SOL_train, name).replace('bg', 'sol')) 
               for name in sample_names_train]

sample_files_valid = [(os.path.join(DIR_BG_valid, name), 
               os.path.join(DIR_SOL_valid, name).replace('bg', 'sol')) 
               for name in sample_names_valid]



train_files = sample_files_train[:]
valid_files = sample_files_valid[:] 

train_data = GraphDataset(train_files)  
train_loader = torch_geometric.loader.DataLoader(train_data, 
                batch_size=BATCH_SIZE, shuffle=True)
valid_data = GraphDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(valid_data,
                batch_size=BATCH_SIZE, shuffle=True)


def train(predict, data_loader, optimizer=None, weight_norm=1):
  
    if optimizer:
        predict.train() 
    else:
        predict.eval()   
        
    mean_loss = 0
    total_correct = 0  
    total_predictions = 0  
    n_samples_processed = 0
    
    with torch.set_grad_enabled(optimizer is not None):  
        for step, batch in enumerate(data_loader):
            batch = batch.to(DEVICE)
           
            solInd = batch.nsols 
            target_sols = []
            target_vals = []
            solEndInd = 0
            valEndInd = 0

           
            for i in range(solInd.shape[0]):
                nvar = len(batch.varInds[i][0][0])  
                
                solStartInd = solEndInd   
                solEndInd = solInd[i] * nvar + solStartInd  
                valStartInd = valEndInd
                valEndInd = valEndInd + solInd[i]
                
                sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
                vals = batch.objVals[valStartInd:valEndInd]
                target_sols.append(sols)
                target_vals.append(vals)

           
            batch.constraint_features[torch.isinf(batch.constraint_features)] = 10 
            BD = predict(
                batch.constraint_features,  
                batch.edge_index,         
                batch.edge_attr,           
                batch.variable_features,  
            )
            BD = BD.sigmoid() 
            loss = 0
            index_arrow = 0 
            
           
            for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
                n_vals = vals[0, :] 

                exp_weight = torch.exp(-n_vals / weight_norm)  
                weight = exp_weight/exp_weight.sum() 
    
                varInds = batch.varInds[ind]
                varname_map = varInds[0][0]
                b_vars=varInds[1][0].long()       
              
                sols = sols[:, varname_map][:, b_vars]
              
                n_var = batch.ntvars[ind]
                
                pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[b_vars]
                predictions = (pre_sols >= 0.5).float() 
               
                correct = (predictions == sols).float().sum()  
                total_correct += correct.item()
                total_predictions += sols.numel()  

                index_arrow += n_var
                
                pos_loss = -(pre_sols + 1e-8).log()[None, :] * (sols==1).float() 
                neg_loss = -(1 - pre_sols + 1e-8).log()[None, :] * (sols==0).float()
                sum_loss = pos_loss + neg_loss
                sample_loss = sum_loss * weight[:, None]
                loss += sample_loss.sum()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                
           
            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs            
    mean_loss /= n_samples_processed 
    accuracy = total_correct / total_predictions * 100  
    
    return mean_loss, accuracy


if __name__ == '__main__':  
    torch.manual_seed(0)  
    PredictModel = GNNPolicy().to(DEVICE)  
    optimizer = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE)

    weight_norm = 10000 
    best_val_loss = 99999 
    best_acc = 0
  
    for epoch in range(NB_EPOCHS):
        begin=time.time()     
        train_loss, train_acc = train(PredictModel, train_loader, optimizer, weight_norm)
        print(f"Epoch {epoch} Train loss: {train_loss:0.6f} Train accuracy: {train_acc:.5f}%")
        st = f'@epoch{epoch} Train loss:{train_loss:.6f} Train acc:{train_acc:.2f}% Valid loss:{valid_loss:.6f} Valid acc:{valid_acc:.2f}% TIME:{time.time()-begin}\n'
        log_file.write(st.encode())
        log_file.flush()
    torch.save(PredictModel.state_dict(), model_save_path+'model_last.pth')
    print('done') 
