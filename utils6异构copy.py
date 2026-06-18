import os
import os.path as osp
import random
import numpy as np
import pandas as pd
import torch
from dataset.PISynergy_dataset import MyInMemoryDataset
from metrics import get_metrics, get_metrics_cf
from dataPreprocess import synergyConvert_id2tensor
from torch.utils.data import TensorDataset
import re
import pickle
from torch_geometric.data import *
from rdkit import Chem
from mol_to_linegraph import *
#from PITSynergy.model.DMPNN堆叠操作 import CoAttentionLayer
from sklearn.metrics import f1_score
import json




def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 保存模型参数，提早结束
class EarlyStopping():
    def __init__(self, mode='higher', patience=50, filename=None, metric=None, n_fold=None, folder=None, model = None):
        """
        Initialize EarlyStopping object.

        Args:
            mode (str): 'higher' if a higher score is better, 'lower' if a lower score is better.
            patience (int): Number of epochs to wait for improvement before early stopping.
            filename (str): Name of the checkpoint file to save the model state.
            metric (str): Metric to monitor for early stopping. Can be 'r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', or 'mse'.
            n_fold (int): Fold number used for naming checkpoint file.
            folder (str): Folder path to save checkpoint file.
        """

        if filename is None:
            filename = os.path.join(folder, '{}_{}_fold_early_stop.pth'.format(model, n_fold))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score' or 'mse', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """
        Check if the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """
        Check if the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):

        model.load_state_dict(torch.load(self.filename))

def load_data(args):
    work_dir = args.workdir
    dataset_index = args.dataset_index
    dataset_name = args.dataset_name
    data_root = osp.join(work_dir,'data')   # ./data

    # 读取细胞系特征矩阵
    celllines_data = None
    if args.celldataset == 1:
        celllines_data = osp.join(data_root,'cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_path0 = osp.join(data_root,f'/home/lkp/cywhome/PITSynergy/PITSynergy/data/independent dataset/{dataset_name}/clines.pt')
        celllines_path = osp.join(data_root,f'/home/lkp/cywhome/PITSynergy/PITSynergy/13/cline_de_exp.pt')
        celllines_data = torch.load(celllines_path, weights_only=False)
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root,'cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes.npy')

    drugs_path = osp.join(data_root,f'/home/lkp/cywhome/PITSynergy/PITSynergy/data/independent dataset/{dataset_name}/drugs.pt')
    drugs_data = torch.load(drugs_path, weights_only=False)

    synergy_data_path = osp.join(data_root, f'/home/lkp/cywhome/PITSynergy/PITSynergy/13/synergyDMPNN2.csv')
 

    synergy_data = pd.read_csv(synergy_data_path)



    synergy_data = synergyConvert_id2tensor(synergy_data, drugs_data, celllines_data)

    print(f'cline data:{len(celllines_data)}')
    print(f'durgs data:{len(drugs_data)}')
    print(f'synergy data:{len(synergy_data)}')
    # 使用drop函数删除第一列
    synergy_data = synergy_data.drop(synergy_data.columns[0], axis=1)
    return synergy_data




def get_TensorDataset(synergy_data):
    # 假设你的特征已经转换为了PyTorch张量
    druga = synergy_data.iloc[:,0].tolist()
    drugb = synergy_data.iloc[:,1].tolist()
    cline = synergy_data.iloc[:,2].tolist()
    score = synergy_data.iloc[:,3].values
    druga = torch.tensor(druga)
    drugb = torch.tensor(drugb)
    cline = torch.tensor(cline)
    score = torch.tensor(score)
    # 创建TensorDataset
    dataset = TensorDataset(druga, drugb, cline, score)
    print("TensorDataset done")
    return dataset


def get_DataList(synergy_data):
    # 假设你的特征已经转换为了PyTorch张量
    druga = synergy_data.iloc[:,0].tolist()
    drugb = synergy_data.iloc[:,1].tolist()
    cline = synergy_data.iloc[:,2].tolist()
    score = synergy_data.iloc[:,3].values

    uida=synergy_data.iloc[:,0].tolist()
    uidb = synergy_data.iloc[:, 1].tolist()

    data_list = list(zip(druga, drugb, cline, score,uida,uidb))#有蛋白的时候这里要加
    print("data_list done")
    return data_list

def parse_custom_data_str(s):
    # 定义正则表达式模式，用于匹配属性和对应的值
    pattern = r'(\w+)=(\[.*?\])'
    matches = re.findall(pattern, s)

    # 创建一个空字典，用于存储属性和对应的值
    attrs = {}
    for attr, value in matches:
        # 将字符串形式的列表转换为 Python 列表
        list_value = eval(value)
        # 根据列表创建对应形状的张量
        tensor_value = torch.zeros(list_value)
        attrs[attr] = tensor_value

    # 使用提取的属性创建 CustomData 对象
    return CustomData(**attrs)

def view_pkl_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print("错误: 文件未找到!")
    except pickle.UnpicklingError:
        print("错误: 无法解包文件，可能不是有效的 pkl 文件。")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return None


def train0(model,criterion,opt,dataloader,device,args=None):
    file_path = r'/home/lkp/cywhome/PITSynergy/PITSynergy/13/drug_data.pkl'
    data = view_pkl_file(file_path)
    model.train()
    train_loss_sum = 0
    batch = 0
    for druga, drugb, cline, score ,uida,uidb in dataloader:
        batch += 1
        model.zero_grad()

        druga_list = []
        drugb_list = []
        uida_list=[]
        uidb_list=[]
        for i, (a, b) in enumerate(zip(druga, drugb)):
            d1 = data[a]
            d2 = data[b]
            # 为每个图对象添加 edge_batch（每条边对应的图编号）
            d1.edge_batch = torch.full((d1.edge_index.size(1),), i, dtype=torch.long)
            d2.edge_batch = torch.full((d2.edge_index.size(1),), i, dtype=torch.long)
            druga_list.append(d1)
            drugb_list.append(d2)
        druga_batch = Batch.from_data_list([data[a] for a in druga], follow_batch=['edge_index']).to(device)
        drugb_batch = Batch.from_data_list([data[b] for b in drugb], follow_batch=['edge_index']).to(device)
        cline = cline.to(device)
        score = score.to(device).to(torch.float32).view(-1, 1)

        output = model(druga_batch, drugb_batch, cline,uida,uidb).view(-1, 1)
        train_loss = criterion(output, score)
        train_loss_sum += train_loss.item()
        train_loss.backward()
        opt.step()
    loss = train_loss_sum / batch
    return loss

import torch
from torch_geometric.data import Batch
import pickle
import pandas as pd
import numpy as np


# 辅助函数：读取并处理 CSV 相似度矩阵 (用于加载全局药物先验)
def load_global_similarity_csv(file_path, device):
    """读取 CSV 文件中的全局相似度矩阵，并直接返回 Tensor。"""
    try:
        # 假设 CSV 有 ID 列 (index_col=0)，但我们只取数据部分
        df = pd.read_csv(file_path, index_col=0) 
    except FileNotFoundError:
        print(f"错误: 找不到全局相似度文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取全局相似度文件 {file_path} 时出错: {e}")
        return None

    # 转换为 PyTorch Tensor
    global_matrix = torch.tensor(df.values, dtype=torch.float32).to(device)
    return global_matrix


def train(model, criterion, opt, dataloader, device, args=None):
    # --- 1. 读取药物图数据 ---
    file_path = r'/home/lkp/cywhome/PITSynergy/PITSynergy/13/drug_data.pkl' 
    data = view_pkl_file(file_path)
    
    # --- 2. 加载全局药物先验相似度矩阵 (drug_sim_mat) ---
    # 【请替换为您的药物 CSV 文件路径】
    file_drug_prior_path = r'/home/lkp/cywhome/PITSynergy/PITSynergy/13/drug_tanimoto_matrix.csv' 
    drug_sim_mat_prior = load_global_similarity_csv(file_drug_prior_path, device)
    
    if drug_sim_mat_prior is None:
        raise ValueError("未能成功加载全局药物相似度矩阵，请检查路径和文件格式。")

    # 假设 loss_func 就是您在公式中使用的 criterion (如 MSELoss)
    loss_func = criterion 
    loss_func1=torch.nn.MSELoss()
    model.train()
    train_loss_sum = 0
    num_batches = len(dataloader) # 获取dataloader的总批次数

    # 定义重构和对比损失的固定权重
    lambda_rec_drug = 0.5
    lambda_cont = 0.5
    
    for druga, drugb, cline, score, uida, uidb in dataloader:
        model.zero_grad()

        # ------------------------
        # | 数据准备 (保持不变) |
        # ------------------------
        druga_batch = Batch.from_data_list([data[a] for a in druga], follow_batch=['edge_index']).to(device)
        drugb_batch = Batch.from_data_list([data[b] for b in drugb], follow_batch=['edge_index']).to(device)

        #cline = [torch.stack(inner_list) for inner_list in cline]
        #cline = torch.stack(cline).to(device)
        cline = cline.to(device)
        #cline = cline.permute(2, 0, 1)
        
        batch_label = score.to(device).to(torch.float32).view(-1, 1)

        pred=model(druga_batch, drugb_batch, cline, uida, uidb)
        pred = pred.view(-1, 1)
        
        # --- 1. 计算各项损失 ---
        loss_pred = loss_func(pred, batch_label)       
        
        total_loss = loss_pred #+ 0* loss_rec_1 
    
        train_loss_sum += total_loss.item() 
        
        total_loss.backward()
        opt.step()
        
        
    # 返回每个 epoch 的平均损失
    return train_loss_sum / num_batches


def validate0(model, dataloader, device, args=None):
    # file_path = r'/home/lkp/cywhome/PITSynergy/PITSynergy/data/independent dataset/indep0-oneil1/oneil36新.pkl'
    file_path = r'/home/lkp/cywhome/PITSynergy/PITSynergy/13/drug_data.pkl'
    data = view_pkl_file(file_path)
    #file_protein_path=r'/home/cyw/PITSynergy3 - 副本/PITSynergy/data/target_protein_esm2_embeddings.pkl'
    #data1=view_pkl_file(file_protein_path)

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for druga, drugb, cline, score,uida,uidb in dataloader:
            druga_list = []
            drugb_list = []
            
            for i, (a, b) in enumerate(zip(druga, drugb)):
                d1 = data[a]
                d2 = data[b]
                d1.edge_batch = torch.full((d1.edge_index.size(1),), i, dtype=torch.long)
                d2.edge_batch = torch.full((d2.edge_index.size(1),), i, dtype=torch.long)
                druga_list.append(d1)
                drugb_list.append(d2)

            druga_batch = Batch.from_data_list([data[a] for a in druga], follow_batch=['edge_index']).to(device)
            drugb_batch = Batch.from_data_list([data[b] for b in drugb], follow_batch=['edge_index']).to(device)

            cline = [torch.stack(inner_list) for inner_list in cline]  # shape: [n, m, dim]
            cline = torch.stack(cline).to(device)
            cline = cline.permute(2, 0, 1)
            score = score.to(device).to(torch.float32).view(-1, 1)

            output= model(druga_batch, drugb_batch, cline,uida,uidb)
            output=output.view(-1, 1)
            y_pred.append(output)
            y_true.append(score)

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()

    if args.is_regression:
        mse, rmse, mae, r2, pearson, spearman = get_metrics(y_true, y_pred)
        m1, m2, m3, m4, m5, m6 = mse, rmse, mae, r2, pearson, spearman
        return m1, m2, m3, m4, m5, m6
    else:
        auc, aupr, acc, precision, recall = get_metrics_cf(y_pred, y_true)

        # 添加 F1 分数计算（需将 y_pred 转换为标签）
        y_pred_label = (y_pred >= 0.5).astype(int)
        y_true_label = y_true.astype(int)
        #auc, aupr, acc, precision, recall = get_metrics_cf(y_pred_label, y_true_label)
        f1 = f1_score(y_true_label, y_pred_label)

        m1, m2, m3, m4, m5, m6 = auc, aupr, acc, precision, recall, f1
        return m1, m2, m3, m4, m5, m6

    '''else:
        auc, aupr, acc, precision, recall = get_metrics_cf(y_pred, y_true)
        m1, m2, m3, m4, m5 = auc, aupr, acc, precision, recall
        return m1, m2, m3, m4, m5'''




import torch
from torch_geometric.data import Batch
import numpy as np
# 假设您在这里使用了 sklearn.metrics.f1_score
from sklearn.metrics import f1_score 

# ----------------------------------------------------------------------
# 注意：view_pkl_file 和 get_metrics_cf 函数假定在外部已定义，此处未包含
# ----------------------------------------------------------------------

def validate(model, dataloader, device, args=None):
    # file_path = r'/home/lkp/cywhome/PITSynergy/PITSynergy/data/independent dataset/indep0-oneil1/oneil36新.pkl'
    file_path = r'/home/lkp/cywhome/PITSynergy/PITSynergy/13/drug_data.pkl'
    data = view_pkl_file(file_path)

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for druga, drugb, cline, score, uida, uidb in dataloader:
            
            # --- 数据准备 ---
            druga_batch = Batch.from_data_list([data[a] for a in druga], follow_batch=['edge_index']).to(device)
            drugb_batch = Batch.from_data_list([data[b] for b in drugb], follow_batch=['edge_index']).to(device)

            # cline = [torch.stack(inner_list) for inner_list in cline]  # shape: [n, m, dim]
            # cline = torch.stack(cline).to(device)
            # cline = cline.permute(2, 0, 1)
            cline = cline.to(device)
            score = score.to(device).to(torch.float32).view(-1, 1)

            # --- 模型前向传播 ---
            model_outputs = model(druga_batch, drugb_batch, cline, uida, uidb)
            # 提取第一个输出项 (Logits)，如果模型返回的是元组
            output= model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
            
            # 形状调整
            output = output.view(-1, 1)
            
            # --- 关键修改：将 Logits 转换为概率 (0 到 1) ---
            output_proba = torch.sigmoid(output) 

            y_pred.append(output)  # 存储概率值 (用于 AUC, AUPR)
            y_true.append(score)

    # --- 结果整合与转换 ---
    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy() # 此时 y_pred 是概率值

    if args.is_regression:
        mse, rmse, mae, r2, pearson, spearman = get_metrics(y_true, y_pred)
        m1, m2, m3, m4, m5, m6 = mse, rmse, mae, r2, pearson, spearman
        return m1, m2, m3, m4, m5, m6
    else:
        # 评估指标计算 (使用概率值 y_pred)
        auc, aupr, acc, precision, recall = get_metrics_cf(y_pred, y_true)

        # 添加 F1 分数计算 (需要硬标签)
        y_pred_label = (y_pred >= 0.5).astype(int)
        y_true_label = y_true.astype(int)
        
        f1 = f1_score(y_true_label, y_pred_label)

        m1, m2, m3, m4, m5, m6 = auc, aupr, acc, precision, recall, f1
        return m1, m2, m3, m4, m5, m6


def validate2(model, dataloader, device, args=None):
    file_path = r'/home/lkp/cywhome/PITSynergy/PITSynergy/13/drug_data.pkl'
    data = view_pkl_file(file_path)

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for druga, drugb, cline, score, uida, uidb in dataloader:

            # -------- 数据准备 --------
            druga_batch = Batch.from_data_list(
                [data[a] for a in druga],
                follow_batch=['edge_index']
            ).to(device)

            drugb_batch = Batch.from_data_list(
                [data[b] for b in drugb],
                follow_batch=['edge_index']
            ).to(device)

            cline = cline.to(device)

            # score：统一当连续值处理
            score = score.to(device).float().view(-1)

            # -------- 前向 --------
            outputs = model(druga_batch, drugb_batch, cline, uida, uidb)
            output = outputs[0] if isinstance(outputs, tuple) else outputs
            output = output.view(-1)   # [N]

            # -------- 分任务处理 --------
            if args.is_regression:
                # 回归：直接用原始输出
                y_pred.append(output)
            else:
                # 分类：logits → probability
                y_pred.append(torch.sigmoid(output))

            y_true.append(score)

    # -------- 汇总 --------
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()

    # -------- 指标 --------
    mse, rmse, mae, r2, pearson, spearman = get_metrics(y_true, y_pred)
    return mse, rmse, mae, r2, pearson, spearman