import numpy as np
import torch
import gurobipy as gp
import re
from collections import Counter

torch.set_printoptions(threshold=10000)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
NODE_PATTERN = re.compile(r'node\s*\[\s*\n\s*id\s*(\d+)\s*\n\s*label\s*\"([^\"]+)\"')
EDGE_PATTERN = re.compile(r'edge\s*\[\s*\n\s*source\s*(\d+)\s*\n\s*target\s*(\d+)')
source = torch.empty(0, dtype=torch.long)
target = torch.empty(0, dtype=torch.long)
s = torch.empty(0, dtype=torch.long)

def prase_topo(file):
    global source, target, s
    with open(file, 'r', encoding='utf-8') as fp:
        content = fp.read()
        
       
        nodes = [{'id': match.group(1)} for match in NODE_PATTERN.finditer(content)]
        
        # edges = []
        edges = list(EDGE_PATTERN.finditer(content))
        source = torch.tensor([int(match.group(1)) for match in edges], device=device)
        target = torch.tensor([int(match.group(2)) for match in edges], device=device)
        
      
        s = torch.cat([source, source, target, target])
        return len(nodes), len(edges)

def position_get_ordered_flt(variable_features):
  
    lens = variable_features.shape[0]
    feature_width = 20
    
    
    sorter = variable_features[:,1]
    position = torch.argsort(sorter) / float(lens)
    
    
    position_feature = torch.zeros((lens, feature_width), device=device)
    
   
    divider = torch.tensor([2.0**(-i) for i in range(feature_width)], device=device)
    remainder = position.unsqueeze(1).repeat(1, feature_width)
    
    position_feature = (remainder // divider) % 2
    
    return torch.cat([variable_features, position_feature], dim=1)
    
def get_BG_from_GRB(ins_name, topo_name):
   
    m = gp.read(ins_name)  
    num_nodes, num_edges  = prase_topo(f'{topo_name}') 

    ori_start = 6  
    ncons = num_edges * 3 + 1

    unique_s, counts = torch.unique(s, return_counts=True)
    sorted_indices = torch.argsort(unique_s)
    sorted_counts = counts[sorted_indices]
    
    
    
    variables = m.getVars()
    variables.sort(key=lambda v: (0 if v.VarName.startswith('l[') else 1, 
                                int(v.VarName.split('[')[1].split(']')[0])))
   
   
    vars2idx = {}
    for idx,v in enumerate(variables):
        vars2idx[v.VarName] = idx


    v_nodes = torch.zeros(num_nodes + num_edges, ori_start, dtype=torch.float32, device=device)
    b_vars = torch.arange(num_nodes + num_edges, dtype=torch.int32, device=device) 
    v_nodes[:, 3] = 0
    v_nodes[:, 4] = 1e+20
    v_nodes[:, ori_start - 1] = 1

    v_nodes[:num_edges, 0] = -1
    v_nodes[num_edges:, 0] = 1
    v_nodes[:num_edges, 1] = 3 / ncons  
    v_nodes[num_edges:, 1] = (-sorted_counts + 1) / ncons
    v_nodes[:num_edges, 2] = 3
    v_nodes[num_edges:, 2] = sorted_counts + 1
    v_nodes[:num_edges, 3:5] = 1
    v_nodes[num_edges:, 3] = 1
    v_nodes[num_edges:, 4] = -1
   
    obj_num = m.NumObj 
    obj_cons = torch.zeros(obj_num, num_edges + num_nodes, device=device)  
    obj_cons[0, num_edges:] = 1
    obj_cons[1, :num_edges] = -1


    obj_node = torch.zeros(obj_num, 4, device=device)
    obj_node[0, 0] = 1
    obj_node[0, 1] = m.getObjective(0).size()
    obj_node[1, 0] = -1
    obj_node[1, 1] = m.getObjective(1).size()

    obj = m.getObjective(2)
    for i in range(obj.size()): 
        vars_name = obj.getVar(i).VarName  
        v = obj.getCoeff(i) 
        v_idx = vars2idx[vars_name]  
        obj_cons[2, v_idx] = v  
        v_nodes[v_idx ,0] += v  
        obj_node[2, 0] += v  
        obj_node[2, 1] += 1  
    obj_node[2, 0] /= obj_node[2, 1] 


    c_nodes = torch.zeros((ncons, 4), dtype=torch.float32, device=device)
    c_nodes[0] = torch.tensor([1, num_nodes, num_nodes, 0], dtype=torch.float32, device=device)
    c_nodes[1:num_edges + 1] = torch.tensor([-1/3, 3, 0, 0], dtype=torch.float32, device=device)
    c_nodes[num_edges + 1:num_edges*3+ 1] = torch.tensor([0, 2, 0, 1], dtype=torch.float32, device=device)
    c_nodes = torch.cat([c_nodes, obj_node], dim=0)

    
    A = torch.zeros((ncons + obj_num, num_nodes + num_edges), device=device)
    A[0, num_edges:] = 1
    tem_n_1 = torch.zeros((num_edges, num_nodes), device=device)
    tem_n_2 = torch.zeros((num_edges, num_nodes), device=device)
    tem_n_3 = torch.zeros((num_edges, num_nodes), device=device)
    tem_c = torch.eye(num_edges, device=device).repeat(3,1)
    tem_n_1[range(num_edges), source] = 1
    tem_n_1[range(num_edges), target] = 1
    tem_n_2[range(num_edges), source] = 1
    tem_n_3[range(num_edges), target] = 1

    tem_A = torch.cat([tem_n_1, tem_n_2, tem_n_3], dim=0)
    tem_A = torch.cat([tem_c, tem_A], dim=1)
    tem_A = torch.cat([tem_A, obj_cons], dim=0)
    A[1:, :] = tem_A

   
    clip_max = torch.tensor([20000, 1, torch.max(v_nodes[:,2]).item()], device=device)
    clip_min = torch.tensor([0, -1, 0], device=device)

    v_nodes[:,0] = torch.clamp(v_nodes[:,0], clip_min[0], clip_max[0]) 
    
   
    maxs = torch.max(v_nodes,0)[0]
    mins = torch.min(v_nodes,0)[0]
    diff = maxs - mins
    diff[diff == 0] = 1  

    # for ks in range(diff.shape[0]):
    #     if diff[ks] == 0:
    #         diff[ks] = 1 
    v_nodes = (v_nodes - mins) / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    
    
    v_nodes = position_get_ordered_flt(v_nodes)

    
    maxs = torch.max(c_nodes,0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)
    return A.to_sparse(), vars2idx, v_nodes, c_nodes, b_vars
