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



test_task=f'test'


BATCH_SIZE = 8      


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DIR_BG_test = f'./dataset/{test_task}/BG'   
DIR_SOL_test = f'./dataset/{test_task}/solution' 
sample_names_test = os.listdir(DIR_BG_test) 



sample_files_test = [(os.path.join(DIR_BG_test, name), 
               os.path.join(DIR_SOL_test, name).replace('bg', 'sol')) 
               for name in sample_names_test]


test_files = sample_files_test[:]


test_data = GraphDataset(test_files) 
test_loader = torch_geometric.loader.DataLoader(test_data, 
                batch_size=BATCH_SIZE)



def test(predict, data_loader):
    predict.eval()  
    total_correct = 0  
    total_predictions = 0  

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

        index_arrow = 0  
        

        for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
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
             
    accuracy = total_correct / total_predictions * 100  
    return accuracy


if __name__ == '__main__':  
    torch.manual_seed(0) 
    PredictModel = GNNPolicy().to(DEVICE)
    PredictModel.load_state_dict(torch.load('./pretrain/train/0902.pth', map_location=DEVICE))  
    # 验证阶段
    begin = time.time()
    valid_acc = test(PredictModel, test_loader)
    print(f'TIme: {time.time()-begin}s')
    print(f"Test accuracy: {valid_acc:.5f}%")


