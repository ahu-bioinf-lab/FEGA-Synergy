import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAT
#from torch_geometric.nn.glob import global_max_pool as gmp # 假设 gmp 是这样导入的
from torch_geometric.nn.glob import global_mean_pool as gmp

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugDrugCoAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, d1, d2):

        q1 = self.Wq(d1)
        k2 = self.Wk(d2)
        v2 = self.Wv(d2)

        score12 = (q1 * k2).sum(dim=1, keepdim=True) * self.scale
        attn12 = torch.softmax(score12, dim=1)

        d1_new = d1 + attn12 * v2


        q2 = self.Wq(d2)
        k1 = self.Wk(d1)
        v1 = self.Wv(d1)

        score21 = (q2 * k1).sum(dim=1, keepdim=True) * self.scale
        attn21 = torch.softmax(score21, dim=1)

        d2_new = d2 + attn21 * v1

        return d1_new, d2_new


class DrugCellCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, drug, cell):

        q = self.Wq(drug)
        k = self.Wk(cell)
        v = self.Wv(cell)

        score = (q * k).sum(dim=1, keepdim=True) * self.scale
        attn = torch.softmax(score, dim=1)

        drug_new = drug + attn * v

        return drug_new
    
from torch_geometric.nn import GCNConv, global_max_pool as gmp


class DeepDDSGCNNet(torch.nn.Module):

    def __init__(self,
                 n_output=1,
                 n_filters=32,
                 embed_dim=128,
                 num_features_xd=64,
                 num_features_xt=1030,
                 output_dim=128,
                 dropout=0.2,
                 args=None):

        super(DeepDDSGCNNet, self).__init__()

        self.args = args
        self.n_output = n_output

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        self.lin = nn.Linear(39, 64)

        # =========================
        # Drug1 GCN
        # =========================

        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.drug1_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)

        self.drug1_fc_g1 = nn.Linear(num_features_xd * 4, num_features_xd * 2)
        self.drug1_fc_g2 = nn.Linear(num_features_xd * 2, output_dim)

        # =========================
        # Drug2 GCN
        # =========================

        self.drug2_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug2_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.drug2_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)

        self.drug2_fc_g1 = nn.Linear(num_features_xd * 4, num_features_xd * 2)
        self.drug2_fc_g2 = nn.Linear(num_features_xd * 2, output_dim)

        # =========================
        # Cell encoder
        # =========================

        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # =========================
        # Attention modules
        # =========================

        self.dd_attn = DrugDrugCoAttention(output_dim)

        self.dc_attn = DrugCellCrossAttention(output_dim)

        # =========================
        # Final predictor
        # =========================

        self.fc1 = nn.Linear(4 * output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

    def forward(self, data1, data2, x_cell, ua=None, ub=None):

        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # =========================
        # Drug1 branch
        # =========================

        x1 = self.lin(x1)

        x1 = self.relu(self.drug1_conv1(x1, edge_index1))
        x1 = self.relu(self.drug1_conv2(x1, edge_index1))
        x1 = self.relu(self.drug1_conv3(x1, edge_index1))

        x1 = gmp(x1, batch1)

        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout_layer(x1)

        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout_layer(x1)

        # =========================
        # Drug2 branch
        # =========================

        x2 = self.lin(x2)

        x2 = self.relu(self.drug2_conv1(x2, edge_index2))
        x2 = self.relu(self.drug2_conv2(x2, edge_index2))
        x2 = self.relu(self.drug2_conv3(x2, edge_index2))

        x2 = gmp(x2, batch2)

        x2 = self.relu(self.drug2_fc_g1(x2))
        x2 = self.dropout_layer(x2)

        x2 = self.drug2_fc_g2(x2)
        x2 = self.dropout_layer(x2)

        # =========================
        # Cell branch
        # =========================

        x_cell = x_cell[:, :, 0]

        cell_vector = F.normalize(x_cell, 2, 1)

        cell_vector = self.reduction(cell_vector)

        # =========================
        # Drug-Drug Attention
        # =========================

        x1, x2 = self.dd_attn(x1, x2)

        # =========================
        # Drug-Cell Attention
        # =========================

        x1 = self.dc_attn(x1, cell_vector)
        x2 = self.dc_attn(x2, cell_vector)

        # =========================
        # Drug pair interaction
        # =========================

        drug_pair = x1 * x2

        # =========================
        # Concatenate
        # =========================

        xc = torch.cat((x1, x2, cell_vector, drug_pair), dim=1)

        xc = self.relu(self.fc1(xc))
        xc = self.dropout_layer(xc)

        xc = self.relu(self.fc2(xc))
        xc = self.dropout_layer(xc)

        out = self.out(xc)

        out = torch.sigmoid(out)

        return out

class DeepDDSGCNNet0(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=64, num_features_xt=1030, output_dim=128, dropout=0.2,args=None):

        super(DeepDDSGCNNet, self).__init__()
        self.args = args
        self.lin=nn.Linear(39,64)

        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout) # 重命名以避免与变量名冲突
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*4, num_features_xd*2)
        self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)

        # SMILES2 graph branch
        # 注意：这里和原始GCNNet一样，共享了drug1的GCN层权重。
        # 如果需要独立的GCN分支，请将这里的 drug1_convX 改为 drug2_convX 并单独初始化
        self.drug2_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug2_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.drug2_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4, num_features_xd*2)
        self.drug2_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)

        self.args = args
        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp':0,'mut':1,'cn':2, 'eff':3, 'dep':4, 'met':5}
        self.in_channel = len(self.include_omic)
        self.genes_nums = 4079
        self.li=nn.Linear(24474,4079)


        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, data1, data2, x_cell,ua,ub): # 增加了 x_cell 作为独立输入
        batch_size = self.args.batch_size
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1=self.lin(x1)
        x2=self.lin(x2)
        
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)       # global max pooling

        # flatten
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout_layer(x1) # 使用重命名后的 dropout_layer
        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout_layer(x1) # 使用重命名后的 dropout_layer


        # deal drug2
        # 注意：这里和原始GCNNet一样，如果希望drug2使用独立的GCN层，请改为 self.drug2_convX
        x2 = self.drug1_conv1(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug1_conv2(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug1_conv3(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2)  # global max pooling

        # flatten
        x2 = self.relu(self.drug1_fc_g1(x2))
        x2 = self.dropout_layer(x2) # 使用重命名后的 dropout_layer
        x2 = self.drug1_fc_g2(x2)
        x2 = self.dropout_layer(x2) # 使用重命名后的 dropout_layer


        # deal cell
        # 使用传入的 x_cell，并假设它已经是最终的 cell 特征向量，
        # 如果 x_cell 还需要进一步处理（如 PISynergynet 中对 x_cell 的 omic 选择、view、CNN/Transformer 处理），
        # 则需要在这里添加相应的逻辑。
        # 这里假设 x_cell 已经是经过处理的、可以直接用于 reduction 的特征。
        #x_cell = x_cell[:,:,[self.omic_dict[i] for i in self.include_omic]] 
        x_cell= x_cell[:, :, 0] # [batch*4079,len(omics)]
        #cell = x_cell.view(batch_size, self.genes_nums, -1).type(torch.float32)
        # cell_embed = cell.reshape(batch_size, -1)
        # cell_embed=self.li(cell_embed)
        cell_vector = F.normalize(x_cell, 2, 1) # 对 x_cell 进行归一化
        cell_vector = self.reduction(cell_vector)

        # concat
        xc = torch.cat((x1, x2, cell_vector), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout_layer(xc) # 使用重命名后的 dropout_layer
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout_layer(xc) # 使用重命名后的 dropout_layer
        out = self.out(xc)
        out = torch.sigmoid(out)
    # *** 调试打印语句结束 ***
        return out
    
    def init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

from torch_geometric.nn import GATConv


# GAT model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=64, n_output=1, num_features_xt=4079, output_dim=128, dropout=0.2, args=None):
        super(GATNet, self).__init__()
        
        # graph drug layers (GATConv)
        # 注意：这里heads=10，GATConv的输出特征是 output_dim * heads
        self.drug1_gcn1 = GATConv(num_features_xd, output_dim, heads=10, dropout=dropout)
        # 第二层GATConv的输入维度是第一层输出维度 * heads
        self.drug1_gcn2 = GATConv(output_dim * 10, output_dim, dropout=dropout) 
        
        # 原始模型中 drug1_fc_g1 输入是 num_features_xd*4，这里output_dim = 128
        # global_max_pool后的特征维度就是 GATConv 最后一层的 output_dim (这里是128)
        self.drug1_fc_g1 = nn.Linear(output_dim, output_dim) 
        self.li=nn.Linear(24474,4079)
        self.lin=nn.Linear(42,64)

        # self.drug1_gcn3 = GATConv(output_dim, output_dim, dropout=dropout) # 原始注释掉的层
        # self.drug1_fc_g2 = nn.Linear(2048, output_dim) # 原始注释掉的层
        # self.filename = file # 注释掉，因为PISynergynet接口中没有file参数

        # DL cell features (reduction network)
        # 根据PISynergynet，num_features_xt 可能是一个较大的值，例如 4079
        # 这里为了兼容，继续使用 num_features_xt 作为输入维度
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * 2), # 匹配 PISynergynet 的 cell_conv 输出或最终维度
            nn.ReLU()
        )

        # combined layers
        # PISynergynet 中是 drug_ca_embed (output_dim) + drug_cb_embed (output_dim) + cell_embed (output_dim)
        # 所以总输入维度是 output_dim * 3
        # 但是您的GATNet原始模型是 output_dim * 4
        # 如果要完全匹配PISynergy的融合逻辑，这里的输入应该是 output_dim * 3
        # 暂时保持 output_dim * 4 以保留原始GATNet结构，但请注意这个差异
        self.fc1 = nn.Linear(output_dim * 4, 2048) 
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output) # n_output 保持不变，通常为2 (logits) 或 1 (概率)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout) # 重命名以避免与 F.dropout 混淆
        self.output_dim = output_dim

    # 以下两个方法 (get_col_index, save_num) 在 GATNet 的核心推理逻辑中不是必需的，
    # 并且在 PISynergynet 的接口中也没有对应，因此建议注释掉或移除，以保持模型清晰。
    # 如果它们是用于调试或可视化，可以在训练脚本中单独调用。
    # def get_col_index(self, x):
    #     row_size = len(x[:, 0])
    #     row = np.zeros(row_size)
    #     col_size = len(x[0, :])
    #     for i in range(col_size):
    #         row[np.argmax(x[:, i])] += 1
    #     return row

    # def save_num(self, d, path):
    #     d = d.cpu().numpy()
    #     ind = self.get_col_index(d)
    #     ind = pd.DataFrame(ind)
    #     ind.to_csv('data/case_study/' + path + '_index.csv', header=0, index=0)
    #     # 下面是load操作
    #     # read_dictionary = np.load('my_file.npy').item()
    #     # d = pd.DataFrame(d)
    #     # d.to_csv('data/result/' + path + '.csv', header=0, index=0)
        self.args = args
        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp':0,'mut':1,'cn':2, 'eff':3, 'dep':4, 'met':5}
        self.in_channel = len(self.include_omic)
        self.genes_nums = 4079


    def forward(self, data1, data2, x_cell): # 统一为 PISynergynet 的 forward 接口
        # 从 data1 中提取 x1, edge_index1, batch1。
        # 注意：原始GATNet从data1取了cell，但PISynergy将cell作为独立参数x_cell传入。
        # 因此，这里从data1中只取药物图相关数据，x_cell则直接使用传入的参数。
        batch_size = self.args.batch_size
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1=self.lin(x1)
        x2=self.lin(x2)
        x1 = self.drug1_gcn1(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.drug1_gcn2(x1, edge_index1) # 注意这里不需要 return_attention_weights
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        # 注释掉调试/可视化代码
        # if len(batch1) <1000:
        #     dt = pd.DataFrame(arr)
        #     dt = np.around(dt, decimals=2)
        #     get_map(dt, 'data/case_study/' + self.filename + '_drug1_att_x1')
        #     p = self.filename + '_drug1'
        #     self.save_num(x1, p)
        
        x1 = gmp(x1, batch1) # global max pooling

        x1 = self.drug1_fc_g1(x1)
        x1 = self.relu(x1)
        # x1 = self.drug1_fc_g2(x1) # 原始注释掉的层
        # x1 = self.relu(x1) # 原始注释掉的层


        # deal drug2 (共享 drug1 的 GCN 层)
        x2 = self.drug1_gcn1(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = self.drug1_gcn2(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        # 注释掉调试/可视化代码
        # if len(batch1) < 1000:
        #     dt = pd.DataFrame(arr)
        #     dt = np.around(dt, decimals=2)
        #     get_map(dt, 'data/case_study/' + self.filename + '_drug2_att_x1')
        #     p = self.filename + '_drug2'
        #     self.save_num(x2, p)

        x2 = gmp(x2, batch2)  # global max pooling

        x2 = self.drug1_fc_g1(x2)
        x2 = self.relu(x2)
        # x2 = self.drug1_fc_g2(x2) # 原始注释掉的层
        # x2 = self.relu(x2) # 原始注释掉的层

        x_cell = x_cell[:,:,[self.omic_dict[i] for i in self.include_omic]]  # [batch*4079,len(omics)]
        cell = x_cell.view(batch_size, self.genes_nums, -1).type(torch.float32)
        cell_embed = cell.reshape(batch_size, -1)
        cell_embed=self.li(cell_embed)
        cell_vector = F.normalize(cell_embed, p=2, dim=1) # PISynergy normalize 是在后面，这里保持GATNet原有顺序
        cell_vector = self.reduction(cell_vector) # cell_vector 形状 (batch_size, output_dim * 2)

        xc = torch.cat((x1, x2, cell_vector), 1) 
        xc = F.normalize(xc, p=2, dim=1) # 原始 GATNet 有这一步

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout_layer(xc) # 使用重命名后的 dropout_layer
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout_layer(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout_layer(xc)
        
        out = self.out(xc) # 最终输出
        out = torch.sigmoid(out)

        # 如果您的任务是二分类并且使用 BCELoss，请在这里添加 sigmoid
        # out = torch.sigmoid(out)

        return out

    # PISynergynet 有 init_weights 方法，GATNet 可以选择添加，但这不是接口强制要求
    # 保持GATNet的简洁性，除非有特殊初始化需求，否则可以不添加
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)