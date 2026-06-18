import torch
import torch.nn as nn
import numpy as np
from .layers.PISynergy_utils import predictor
from model.model_utiles import Embeddings, predictor, drug_feat
from torch_geometric.nn import global_mean_pool as gmp

class DeepSynergy(torch.nn.Module):

    def __init__(self,
                 num_attention_heads = 8,
                 attention_probs_dropout_prob = 0.1,
                 hidden_dropout_prob = 0.1,
                 max_length = 50,
                 input_dim_drug=2586,   # 论文是2700个有价值的子结构，这里是2586个，可能删减了
                 output_dim=2560,
                 args=None):
        super(DeepSynergy, self).__init__()

        self.args = args
        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp':0,'mut':1,'cn':2, 'eff':3, 'dep':4, 'met':5}
        self.in_channel = len(self.include_omic)
        self.max_length = max_length
        self.li=nn.Linear(43,200)
        self.lic = nn.Sequential(
                nn.Linear(24474, 16384),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16384, 4079)
            )


        if args.celldataset == 0 :
            self.genes_nums = 697
        elif args.celldataset == 1:
            self.genes_nums = 18498
        elif args.celldataset == 2:
            self.genes_nums = 4079

        # cnn 和 mlp 两种方式处理多组学特征
        hidden_size = 4
        self.patch = 50
        self.drug_emb = Embeddings(input_dim_drug, hidden_size, self.patch, hidden_dropout_prob)
        self.predictor = predictor(hidden_size * self.patch * 2 + 4079) # 两个药物和一个细胞系

    def forward(self, drugA, drugB, x_cell):

        batch_size = self.args.batch_size
        x1, edge_index1, batch1 = drugA.x, drugA.edge_index, drugA.batch
        x2, edge_index2, batch2 = drugB.x, drugB.edge_index, drugB.batch

        # 药物特征提取
        # [bs,2,50] -> [bs,165]  这里主要功能就是填充0
        drugA=drugA.x
        drugB=drugB.x
        drugA=self.li(drugA)
        drugB=self.li(drugB)
        drugA = gmp(drugA, batch1)
        drugB = gmp(drugB, batch2)
        drugA = drugA.float()
        drugB = drugB.float()

        # 细胞系特征提取
        x_cell = x_cell[:,:,[self.omic_dict[i] for i in self.include_omic]]  # [batch*4079,len(omics)]
        cell = x_cell.view(batch_size, self.genes_nums, -1).type(torch.float32)

        drugA_embed = drugA.reshape(batch_size, -1)
        drugB_embed = drugB.reshape(batch_size, -1)
        cell_embed = cell.reshape(batch_size, -1)
        cell_embed=self.lic(cell_embed)
        drug_embed = torch.cat((drugA_embed,drugB_embed),1)
        output = self.predictor(cell_embed, drug_embed)

        return output


    def init_weights(self):

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
