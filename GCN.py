import torch
import torch_geometric

class GNNPolicy(torch.nn.Module):
    """图神经网络策略模型，用于处理二分图数据"""
    
    def __init__(self):
        """初始化模型结构"""
        super().__init__()  # 调用父类构造函数
        emb_size = 64 # 嵌入维度
        cons_nfeats = 4  # 约束节点特征维度
        edge_nfeats = 1  # 边特征维度
        var_nfeats = 26  # 变量节点特征维度、后续可能需要调整
        # 约束节点嵌入层，输出64维
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),  # 层归一化
            torch.nn.Linear(cons_nfeats, emb_size),  # 线性变换
            torch.nn.ReLU(),  # ReLU激活函数
            torch.nn.Linear(emb_size, emb_size),  # 第二层线性变换
            torch.nn.ReLU(),  # ReLU激活函数
        )
        # 边特征嵌入层
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),  # 仅进行层归一化
        )
        # 变量节点嵌入层，输出64维
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),  # 层归一化
            torch.nn.Linear(var_nfeats, emb_size),  # 线性变换
            torch.nn.ReLU(),  # ReLU激活函数
            torch.nn.Linear(emb_size, emb_size),  # 第二层线性变换
            torch.nn.ReLU(),  # ReLU激活函数
        )
        # 定义四个图卷积层
        self.conv_v_to_c = BipartiteGraphConvolution()  # 变量到约束的卷积
        self.conv_c_to_v = BipartiteGraphConvolution()  # 约束到变量的卷积

        self.conv_v_to_c2 = BipartiteGraphConvolution()  # 第二层变量到约束
        self.conv_c_to_v2 = BipartiteGraphConvolution()  # 第二层约束到变量

        # 输出层
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),  # 线性变换
            # torch.nn.Dropout(0.3),
            torch.nn.ReLU(),  # ReLU激活
            torch.nn.Linear(emb_size, 1, bias=False),  # 输出单值预测
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        """前向传播"""
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)  # 反转边方向
        # 第一步：将各特征嵌入到统一维度
        constraint_features = self.cons_embedding(constraint_features)  # 约束特征嵌入
        edge_features = self.edge_embedding(edge_features)  # 边特征嵌入
        variable_features = self.var_embedding(variable_features)  # 变量特征嵌入

        # 两层图卷积
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

        # 最终输出预测
        output = self.output_module(variable_features).squeeze(-1)  # 压缩维度
        return output

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """二分图卷积层, 继承自MessagePassing"""
    
    def __init__(self):
        """初始化卷积层"""
        super().__init__("add")  # 使用加法聚合消息
        emb_size = 64 # 嵌入维度

        # 定义各特征变换模块
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)  # 左侧节点特征变换
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)  # 边特征变换
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)  # 右侧节点特征变换
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),  # 层归一化
            torch.nn.ReLU(),  # ReLU激活
            torch.nn.Linear(emb_size, emb_size),  # 最终变换
        )
        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # 输出层
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),  # 合并特征
            torch.nn.ReLU(),  # ReLU激活
            torch.nn.Linear(emb_size, emb_size),  # 最终输出
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """前向传播"""
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
            self.feature_module_left(node_features_i)  # 源节点特征
            + self.feature_module_edge(edge_features)  # 边特征
            + self.feature_module_right(node_features_j)  # 目标节点特征
        )
        return output
