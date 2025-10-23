import torch
import torch_geometric

class GNNPolicy(torch.nn.Module):
    
    def __init__(self):
        """初始化模型结构"""
        super().__init__() 
        emb_size = 64
        cons_nfeats = 4 
        edge_nfeats = 1  
        var_nfeats = 26 
       
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),  # ReLU激活函数
            torch.nn.Linear(emb_size, emb_size), 
            torch.nn.ReLU(), 
        )
       
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats), 
        )
       
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),  
            torch.nn.Linear(var_nfeats, emb_size), 
            torch.nn.ReLU(), 
            torch.nn.Linear(emb_size, emb_size), 
            torch.nn.ReLU(),  
        )
        # 定义四个图卷积层
        self.conv_v_to_c = BipartiteGraphConvolution()  
        self.conv_c_to_v = BipartiteGraphConvolution()  

        self.conv_v_to_c2 = BipartiteGraphConvolution() 
        self.conv_c_to_v2 = BipartiteGraphConvolution()  

        # 输出层
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size), 
            torch.nn.ReLU(),  
            torch.nn.Linear(emb_size, 1, bias=False), 
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0) 
        constraint_features = self.cons_embedding(constraint_features) 
        edge_features = self.edge_embedding(edge_features) 
        variable_features = self.var_embedding(variable_features) 

      
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )
        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

     
        output = self.output_module(variable_features).squeeze(-1)  
        return output

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    
    
    def __init__(self):
      
        super().__init__("add")  
        emb_size = 64 

      
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size) 
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)  
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False) 
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(), 
            torch.nn.Linear(emb_size, emb_size),  
        )
        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

     
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size), 
            torch.nn.ReLU(), 
            torch.nn.Linear(emb_size, emb_size), 
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
       
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        """消息传递函数"""
        output = self.feature_module_final(
            self.feature_module_left(node_features_i) 
            + self.feature_module_edge(edge_features) 
            + self.feature_module_right(node_features_j) 
        )
        return output

