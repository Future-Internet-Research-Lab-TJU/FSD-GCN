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

# 设置模型和日志保存路径    
model_save_path = f'./pretrain/{train_task}/'
log_save_path = f"train_logs/{train_task}/"
log_file = open(f'{log_save_path}{train_task}_train.log', 'wb')  # 打开日志文件

# 设置训练参数
LEARNING_RATE = 0.001  # 学习率
NB_EPOCHS = 10000  # 训练轮数
BATCH_SIZE = 8        # 批大小
WEIGHT_NORM = 100      # 权重归一化参数

# 设置计算设备（CPU/GPU）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
DIR_BG_train = f'./dataset/{train_task}/BG'      # 二分图数据目录
DIR_SOL_train = f'./dataset/{train_task}/solution'  # 解决方案目录
sample_names_train = os.listdir(DIR_BG_train)  # 获取所有样本文件名

DIR_BG_valid = f'./dataset/{valid_task}/BG'      # 二分图数据目录
DIR_SOL_valid = f'./dataset/{valid_task}/solution'  # 解决方案目录
sample_names_valid = os.listdir(DIR_BG_valid)  # 获取所有样本文件名

# 创建(二分图路径, 解决方案路径)的元组列表
sample_files_train = [(os.path.join(DIR_BG_train, name), 
               os.path.join(DIR_SOL_train, name).replace('bg', 'sol')) 
               for name in sample_names_train]

sample_files_valid = [(os.path.join(DIR_BG_valid, name), 
               os.path.join(DIR_SOL_valid, name).replace('bg', 'sol')) 
               for name in sample_names_valid]


# 划分训练集和验证集
train_files = sample_files_train[:]
valid_files = sample_files_valid[:] 
# 创建数据加载器
train_data = GraphDataset(train_files)  # 生成二部图结构，包含属性看GCN.py注释
train_loader = torch_geometric.loader.DataLoader(train_data, 
                batch_size=BATCH_SIZE, shuffle=True)
valid_data = GraphDataset(valid_files)
valid_loader = torch_geometric.loader.DataLoader(valid_data,
                batch_size=BATCH_SIZE, shuffle=True)


def train(predict, data_loader, optimizer=None, weight_norm=1):
    """
    训练/验证函数
    Args:
        predict: 预测模型
        data_loader: 数据加载器
        optimizer: 优化器(训练时传入)
        weight_norm: 权重归一化参数
    Returns:
        平均损失值
    """
    # 设置模型模式
    if optimizer:
        predict.train()  # 训练模式
    else:
        predict.eval()   # 评估模式
        
    mean_loss = 0
    total_correct = 0  # 正确预测的总数
    total_predictions = 0  # 预测总数
    n_samples_processed = 0
    
    with torch.set_grad_enabled(optimizer is not None):  # 控制梯度计算
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
            loss = 0
            index_arrow = 0  # 跟踪当前处理的变量索引
            
            # 遍历每个样本的解决方案和目标值
            for ind, (sols, vals) in enumerate(zip(target_sols, target_vals)):
                n_vals = vals[0, :]  # 目标值

                exp_weight = torch.exp(-n_vals / weight_norm)  
                weight = exp_weight/exp_weight.sum()  # 归一化权重
    
                varInds = batch.varInds[ind]
                varname_map = varInds[0][0]
                b_vars=varInds[1][0].long()  # 二元变量索引        
                # 提取二元变量的解决方案
                sols = sols[:, varname_map][:, b_vars]
                # 计算交叉熵损失
                n_var = batch.ntvars[ind]
                # print(nvar)
                pre_sols = BD[index_arrow:index_arrow + n_var].squeeze()[b_vars]
                predictions = (pre_sols >= 0.5).float()  # 将概率转换为0/1预测
                # if optimizer is None:
                #     print('sols: ', sols)
                #     print('pre_sols: ', pre_sols)
                # # 计算准确率
                
                correct = (predictions == sols).float().sum()  # 计算正确预测数
                total_correct += correct.item()
                total_predictions += sols.numel()  # 累加预测总数

                index_arrow += n_var
                # 正样本和负样本的损失
                pos_loss = -(pre_sols + 1e-8).log()[None, :] * (sols==1).float() 
                neg_loss = -(1 - pre_sols + 1e-8).log()[None, :] * (sols==0).float()
                sum_loss = pos_loss + neg_loss
                sample_loss = sum_loss * weight[:, None]
                loss += sample_loss.sum()

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                
            # 统计损失
            mean_loss += loss.item()
            n_samples_processed += batch.num_graphs            
    mean_loss /= n_samples_processed  # 计算平均损失
    accuracy = total_correct / total_predictions * 100  # 计算准确率百分比
    
    return mean_loss, accuracy


if __name__ == '__main__':  
    torch.manual_seed(0)  # 设置随机种子
    PredictModel = GNNPolicy().to(DEVICE)  # 将模型转移到设备
    optimizer = torch.optim.Adam(PredictModel.parameters(), lr=LEARNING_RATE)

    weight_norm = 10000 # 获取权重归一化值
    best_val_loss = 99999  # 初始化最佳验证损失
    best_acc = 0
    # 训练循环
    for epoch in range(NB_EPOCHS):
        begin=time.time()
        
        # 训练阶段
        train_loss, train_acc = train(PredictModel, train_loader, optimizer, weight_norm)
        # print(f"Epoch {epoch} Train loss: {train_loss:0.6f} Train accuracy: {train_acc:.5f}%")
        
        # 验证阶段
        valid_loss, valid_acc = train(PredictModel, valid_loader, None, weight_norm)
        print(f"Epoch {epoch} Valid loss: {valid_loss:0.6f} Valid accuracy: {valid_acc:.5f}%")
        # 保存最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            print("保存最佳模型")
            torch.save(PredictModel.state_dict(), model_save_path+'model_best.pth')
        
        # 保存最新模型
        torch.save(PredictModel.state_dict(), model_save_path+'model_last.pth')
        # 改为按epoch更新
        # 记录日志
        st = f'@epoch{epoch} Train loss:{train_loss:.6f} Train acc:{train_acc:.2f}% Valid loss:{valid_loss:.6f} Valid acc:{valid_acc:.2f}% TIME:{time.time()-begin}\n'
        log_file.write(st.encode())
        log_file.flush()
    print('done')  # 训练完成
