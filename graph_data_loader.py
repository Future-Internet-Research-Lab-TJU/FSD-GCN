import torch
import torch_geometric
import pickle
import numpy as np
import random
class GraphDataset(torch_geometric.data.Dataset):
    """图数据集类，用于加载和处理图数据"""
    
    def __init__(self, sample_files, augment=False):
        """初始化数据集"""
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files  # 样本文件列表
        self.augment = augment  # 是否启用数据增强

    def len(self):
        """返回数据集大小"""
        return len(self.sample_files)

    def process_sample(self, filepath):
        """处理单个样本文件"""
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)  # 加载二分图数据
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)  # 加载解决方案数据

        BG = bgData  # 二分图数据
        varNames = solData['var_names']  # 变量名列表
        sols = solData['sols'][:]  # 取前50个解
        objs = solData['objs'][:]  # 取前50个目标值
        # sols = np.round(sols, 0)  # 四舍五入解为整数
        # print('BG: ', BG)
        # print('varNames: ', varNames)
        # print('objs: ', objs)
        # print('sols: ', sols)
        # print('-----------------------------')
        return BG, sols, objs, varNames

    def add_noise(self, features, noise_level=0.1):
        """添加高斯噪声"""
        noise = torch.randn_like(features) * noise_level
        return features + noise
    
    def random_mask(self, features, mask_ratio=0.1):
        """随机遮蔽特征"""
        mask = torch.rand_like(features) > mask_ratio
        return features * mask

    def get(self, index):
        """获取单个样本"""
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])
        A, v_map, v_nodes, c_nodes, b_vars = BG  # 解包二分图数据
        # 准备图数据特征
        constraint_features = c_nodes  # 约束节点特征
        edge_indices = A._indices()  # 边索引，A系数矩阵的索引
        variable_features = v_nodes  # 变量节点特征
        edge_features = A._values().unsqueeze(1)  # 边特征
        edge_features = torch.ones(edge_features.shape)  # 设为全1，因为edge_features是从邻接矩阵的非零元素索引中提取的，目标函数的系数也设置为1
        if self.augment:
            # 对变量特征进行增强
            if random.random() < 0.5:
                variable_features = self.add_noise(variable_features)
            else:
                variable_features = self.random_mask(variable_features)
                
            # 对约束特征进行增强
            if random.random() < 0.5:
                constraint_features = self.add_noise(constraint_features)
            else:
                constraint_features = self.random_mask(constraint_features)
        # print('-----------------------------')
        constraint_features[torch.isnan(constraint_features)] = 1  # 处理NaN值

        # print('-----------------------------')
        # print('constraint_features: ', constraint_features)
        # print('edge_indices: ', edge_indices)
        # print('variable_features: ', variable_features)
        # print('edge_features: ', edge_features)
        # print('constraint_features: ', constraint_features)
        # 创建图数据对象

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            torch.FloatTensor(variable_features.cpu()),
        )
        # 设置图属性
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0] # 节点数量
        graph.solutions = torch.FloatTensor(sols).reshape(-1)  # 解向量
        graph.objVals = torch.FloatTensor(objs)  # 目标值
        graph.nsols = sols.shape[0]  # 解数量
        graph.ntvars = variable_features.shape[0]  # 变量数量
        graph.varNames = varNames  # 变量名列表
        # print('-----------------------------')
        # print('graph.num_nodes: ', graph.num_nodes)
        # print('graph.solutions: ', graph.solutions)
        # print('graph.objVals: ', graph.objVals)
        # print('graph.nsols: ', graph.nsols)
        # print('graph.ntvars: ', graph.ntvars)
        # print('graph.varNames: ', graph.varNames)
        
        # 创建变量名映射
        varname_dict = {}
        varname_map = []
        # i=0
        # for iter in varNames:
        #     varname_dict[iter]=i
        #     i+=1
        # for iter in v_map:
        #     varname_map.append(varname_dict[iter])


        # varname_map=torch.tensor(varname_map)
        n_v = 0  
        for i, name in enumerate(varNames):
            varname_dict[name] = i  # 变量名到索引的映射
        for name in v_map:
            varname_map.append(varname_dict[name]) 
        for i, name in enumerate(varNames):
            if name.startswith('v['):
                n_v += 1
                # print(f"Variable {name} has index {varname_dict[name]}") 
        # print('n_v: ', n_v)
        b_vars = b_vars[len(b_vars) - n_v:]  # 提取二元变量索引
        # print(b_vars)
        varname_map=torch.tensor(varname_map)    
        graph.varInds = [[varname_map.detach().clone()], [b_vars.detach().clone()]]
        # print('-----------------------------')
        # print('varname_dict: ', varname_dict)
        # print('varname_map: ', varname_map)
        # print('graph.varInds: ', graph.varInds)
        '''
        返回图结构，其中图结构包含如下属性：
        - graph.constraint_features: 约束特征(包含目标函数和约束条件)
        - graph.edge_index: 边索引
        - graph.edge_attr: 边特征
        - graph.variable_features: 变量特征
        - graph.num_nodes: 图的总节点数
        - graph.solutions: 所有解的向量
        - graph.objVals: 所有目标值
        - graph.nsols: 解的数量
        - graph.ntvars: 变量的数量
        - graph.varNames: 变量名列表
        - graph.varInds: 变量索引，二元变量索引
        '''
        return graph

class BipartiteNodeData(torch_geometric.data.Data):
    """二分图节点数据类，继承自PyG的Data类"""
    
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features):
        """初始化图数据"""
        super().__init__()
        self.constraint_features = constraint_features  # 约束特征
        self.edge_index = edge_indices  # 边索引
        self.edge_attr = edge_features  # 边特征
        self.variable_features = variable_features  # 变量特征

    def __inc__(self, key, value, store, *args, **kwargs):
        """处理图拼接时的索引增量"""
        if key == "edge_index":  
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)