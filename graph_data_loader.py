import torch
import torch_geometric
import pickle
import numpy as np
import random
class GraphDataset(torch_geometric.data.Dataset):
    
    
    def __init__(self, sample_files, augment=False):
       
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files  
        self.augment = augment 
    def len(self):
        
        return len(self.sample_files)

    def process_sample(self, filepath):
       
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)  
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)  

        BG = bgData  
        varNames = solData['var_names']  
        sols = solData['sols'][:] 
        objs = solData['objs'][:] 
        # sols = np.round(sols, 0)  
        # print('BG: ', BG)
        # print('varNames: ', varNames)
        # print('objs: ', objs)
        # print('sols: ', sols)
        # print('-----------------------------')
        return BG, sols, objs, varNames

    def add_noise(self, features, noise_level=0.1):
      
        noise = torch.randn_like(features) * noise_level
        return features + noise
    
    def random_mask(self, features, mask_ratio=0.1):
       
        mask = torch.rand_like(features) > mask_ratio
        return features * mask

    def get(self, index):
      
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])
        A, v_map, v_nodes, c_nodes, b_vars = BG
       
        constraint_features = c_nodes  
        edge_indices = A._indices() 
        variable_features = v_nodes 
        edge_features = A._values().unsqueeze(1) 
        edge_features = torch.ones(edge_features.shape)  
        if self.augment:
          
            if random.random() < 0.5:
                variable_features = self.add_noise(variable_features)
            else:
                variable_features = self.random_mask(variable_features)
                
          
            if random.random() < 0.5:
                constraint_features = self.add_noise(constraint_features)
            else:
                constraint_features = self.random_mask(constraint_features)
        constraint_features[torch.isnan(constraint_features)] = 1
        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            torch.FloatTensor(variable_features.cpu()),
        )
       
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0] 
        graph.solutions = torch.FloatTensor(sols).reshape(-1)
        graph.objVals = torch.FloatTensor(objs) 
        graph.nsols = sols.shape[0] 
        graph.ntvars = variable_features.shape[0]  
        graph.varNames = varNames 
        
     
        varname_dict = {}
        varname_map = []
      


        n_v = 0  
        for i, name in enumerate(varNames):
            varname_dict[name] = i 
        for name in v_map:
            varname_map.append(varname_dict[name]) 
        for i, name in enumerate(varNames):
            if name.startswith('v['):
                n_v += 1
            
        b_vars = b_vars[len(b_vars) - n_v:]

        varname_map=torch.tensor(varname_map)    
        graph.varInds = [[varname_map.detach().clone()], [b_vars.detach().clone()]]
        return graph

class BipartiteNodeData(torch_geometric.data.Data):
   
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features):
       
        super().__init__()
        self.constraint_features = constraint_features  
        self.edge_index = edge_indices 
        self.edge_attr = edge_features  
        self.variable_features = variable_features  

    def __inc__(self, key, value, store, *args, **kwargs):
       
        if key == "edge_index":  
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == "candidates":
            return self.variable_features.size(0)
        else:

            return super().__inc__(key, value, *args, **kwargs)
