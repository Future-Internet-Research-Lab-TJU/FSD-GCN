import os
import torch
import torch_geometric
import random
import time
import numpy as np
# 导入自定义的图数据集和GNN模型
from graph_data_loader import GraphDataset
from GCN import GNNPolicy

torch.cuda.empty_cache()
# 启用CUDA加速和基准优化
torch.backends.cudnn.enabled = True  
torch.backends.cudnn.benchmark = True 

# 创建训练日志和模型保存目录

test_task=f'test'

# 设置训练参数
BATCH_SIZE = 8        # 批大小

# 设置计算设备（CPU/GPU）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
DIR_BG_test = f'./dataset/{test_task}/BG'      # 二分图数据目录
DIR_SOL_test = f'./dataset/{test_task}/solution'  # 解决方案目录
sample_names_test = os.listdir(DIR_BG_test)  # 获取所有样本文件名


# 创建(二分图路径, 解决方案路径)的元组列表
sample_files_test = [(os.path.join(DIR_BG_test, name), 
               os.path.join(DIR_SOL_test, name).replace('bg', 'sol')) 
               for name in sample_names_test]


test_files = sample_files_test[:]

# 创建数据加载器
test_data = GraphDataset(test_files)  # 生成二部图结构，包含属性看GCN.py注释
test_loader = torch_geometric.loader.DataLoader(test_data, 
                batch_size=BATCH_SIZE)



def test(predict, data_loader):
    predict.eval()   # 评估模式
    total_correct = 0  # 正确预测的总数
    total_predictions = 0  # 预测总数

    for step, batch in enumerate(data_loader):
        batch = batch.to(DEVICE)
        # 解析批数据中的解决方案
        solInd = batch.nsols  # 每个样本的解决方案数量
        target_sols = []
        target_vals = []
        solEndInd = 0
        valEndInd = 0

        # 遍历批中的每个样本
        for i in range(solInd.shape[0]):
            nvar = len(batch.varInds[i][0][0])  # 变量数量
            # 计算解决方案的起止索引
            solStartInd = solEndInd   
            solEndInd = solInd[i] * nvar + solStartInd  
            valStartInd = valEndInd
            valEndInd = valEndInd + solInd[i]
            # 获取解决方案和目标值
            sols = batch.solutions[solStartInd:solEndInd].reshape(-1, nvar)
            vals = batch.objVals[valStartInd:valEndInd]
            target_sols.append(sols)
            target_vals.append(vals)

        # 预测二元变量分布 (Binary Distribution)
        batch.constraint_features[torch.isinf(batch.constraint_features)] = 10  # 处理无穷大值
        BD = predict(
            batch.constraint_features,  # 约束特征
            batch.edge_index,           # 边索引
            batch.edge_attr,            # 边属性
            batch.variable_features,    # 变量特征
        )
        BD = BD.sigmoid()  # 使用sigmoid激活

        index_arrow = 0  # 跟踪当前处理的变量索引
        
        # 遍历每个样本的解决方案和目标值
        for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
            varInds = batch.varInds[ind]
            varname_map = varInds[0][0]
            b_vars=varInds[1][0].long()  # 二元变量索引        
            # 提取二元变量的解决方案
            sols = sols[:, varname_map][:, b_vars]
            # 计算交叉熵损失
            n_var = batch.ntvars[ind]
            pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[b_vars]
            predictions = (pre_sols >= 0.5).float()  # 将概率转换为0/1预测
            # print('pre_sols: ', pre_sols)
            # # 计算准确率
            correct = (predictions == sols).float().sum()  # 计算正确预测数
            total_correct += correct.item()
            total_predictions += sols.numel()  # 累加预测总数

            index_arrow += n_var
             
    accuracy = total_correct / total_predictions * 100  # 计算准确率百分比
    return accuracy


if __name__ == '__main__':  
    torch.manual_seed(0)  # 设置随机种子
    PredictModel = GNNPolicy().to(DEVICE)  # 将模型转移到设备
    PredictModel.load_state_dict(torch.load('./pretrain/train/0902.pth', map_location=DEVICE))  # 加载预训练模型
    # 验证阶段
    begin = time.time()
    valid_acc = test(PredictModel, test_loader)
    print(f'TIme: {time.time()-begin}s')
    print(f"Test accuracy: {valid_acc:.5f}%")

