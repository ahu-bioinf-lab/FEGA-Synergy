import torch
import torch.nn as nn
from .layers.PISynergy_utils import *
from .model_utiles import CellCNN, Embeddings,CellEncoder_CNN_Transformer
from collections import defaultdict
import pandas as pd
from .ESA import *
import json

class FEGA_Synergy(torch.nn.Module):

    def __init__(self,
                 n_layers=1,
                 n=1,
                 hidden_dim=192,
                 attention_dropout_prob=0.2,
                 max_length=50,
                 output_dim=2560,
                 args=None,
                 hetero_data=None,
                 drug2id=None,
                 mask_file=None):
        super(FEGA_Synergy, self).__init__()

        self.args = args
        self.hetero_data = hetero_data
        node_num_dict = {'drug': len(drug2id), 'protein': 810}  # 1027
        feat_dim_dict = {'drug': 64, 'protein': 256}

        self.SEGAHGT = SEGAHGTEncoder(
            dim_drug=300,
            dim_prot=256,
            dim_pathway=768,
            dim_hidden=hidden_dim,
            num_heads=4,
            metadata=hetero_data.metadata(),
            node_num_dict=node_num_dict,
            feat_dim_dict=feat_dim_dict,
            dropout=0.3)

        self.mask_dict = None
        if mask_file is not None:
            df = pd.read_csv(mask_file)
            self.mask_dict = {
                (row["drug_smi"], row["Entry_ID"]): float(row["mask"])
                for _, row in df.iterrows()
            }
            print(f"[INFO] 掩码文件加载完成，共 {len(self.mask_dict)} 条记录。")

        self.drug2id = drug2id

        self.include_omic = args.omic.split(',')
        self.omic_dict = {'exp': 0, 'mut': 1, 'cn': 2, 'eff': 3, 'dep': 4, 'met': 5}
        self.in_channel = len(self.include_omic)
        self.max_length = max_length

        self.noise_layer = FeatureNoise(sigma=0.05)  # 噪声

        if args.celldataset == 0:
            self.genes_nums = 697
        elif args.celldataset == 1:
            self.genes_nums = 18498
        elif args.celldataset == 2:
            self.genes_nums = 4079

        self.fusion=Fusion(hidden_dim)

        self.drug_fc = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
        )

        self.cell2=nn.Sequential(
            nn.Linear(1030,512),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.predictor = predictor(512 * 3)

        self.drug=nn.Linear(300,hidden_dim)

    def forward(self, drugA_graphs, drugB_graphs, x_cell, drugA_smiles, drugB_smiles):
        batch_size = self.args.batch_size
        device = next(self.parameters()).device  # 保证在同一 device

        self.hetero_data = self.hetero_data.to(device)

        # ---- 1. 拿到 dti 边 ----
        dti_edge_index = self.hetero_data['drug', 'interacts', 'protein'].edge_index
        dti_edge_attr = self.hetero_data['drug', 'interacts', 'protein'].edge_attr
        prot_copy2ori = self.hetero_data['protein'].prot_copy2ori
        ppi_edge_index=self.hetero_data['protein','ppi','protein'].edge_index
        #dti_edge_path_id = self.hetero_data['drug', 'interacts', 'protein'].edge_path_id

        drug_embeddings = self.hetero_data['drug'].x.to(device)
        

        # # ---- 3. 取 batch 内药物 embedding ----
        drugA_repr, drugB_repr = [], []
        for i in range(batch_size):
            idx_a = self.drug2id[drugA_smiles[i]]
            idx_b = self.drug2id[drugB_smiles[i]]
            drugA_repr.append(drug_embeddings[idx_a])
            drugB_repr.append(drug_embeddings[idx_b])

        drugA_repr = torch.stack(drugA_repr)  # [batch, feat_dim]
        drugB_repr = torch.stack(drugB_repr)

        # ---- 4. 送进层级 SEGA ----
        drug_structure_repr, drug_semantic_repr = self.SEGAHGT(
            x_dict=self.hetero_data.x_dict,
            dti_edge_index=dti_edge_index,
            dti_edge_attr=dti_edge_attr,
            prot_copy2ori=prot_copy2ori,
            ppi_edge_index=ppi_edge_index,
            #dti_edge_path_id=dti_edge_path_id
        )

        # ---- 5. 取 batch 内药物 SEGA 表示 ----
        drugA_struct_repr, drugB_struct_repr = [], []
        drugA_sema_repr, drugB_sema_repr = [], []

        for i in range(batch_size):
            id_A = self.drug2id[drugA_smiles[i]]
            id_B = self.drug2id[drugB_smiles[i]]

            drugA_struct_repr.append(drug_structure_repr[id_A])
            drugB_struct_repr.append(drug_structure_repr[id_B])
            drugA_sema_repr.append(drug_semantic_repr[id_A])
            drugB_sema_repr.append(drug_semantic_repr[id_B])

        drugA_struct_repr = torch.stack(drugA_struct_repr)
        drugB_struct_repr = torch.stack(drugB_struct_repr)
        drugA_sema_repr = torch.stack(drugA_sema_repr)
        drugB_sema_repr = torch.stack(drugB_sema_repr)


        drug_sim=F.normalize(drug_semantic_repr,2,1)
        drug_cos=torch.mm(drug_sim,drug_sim.T)

        drugA_struct_repr=self.drug(drugA_struct_repr)
        drugB_struct_repr=self.drug(drugB_struct_repr)

        drugA_final = self.fusion(drugA_struct_repr, drugA_sema_repr)
        drugB_final = self.fusion(drugB_struct_repr, drugB_sema_repr)

    
        cell = x_cell[:, :, 0] * x_cell[:, :, 1]
        cell_embed=self.cell2(cell)
    
        drug_ca_embed = self.drug_fc(drugA_final.reshape(batch_size, -1))
        drug_cb_embed = self.drug_fc(drugB_final.reshape(batch_size, -1))

        drug_embed = torch.cat((drug_ca_embed, drug_cb_embed), 1)
        output = self.predictor(cell_embed, drug_embed)

        return output#,drug_cos

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


class FeatureNoise(nn.Module):
    def __init__(self, sigma=0.05):
        super(FeatureNoise, self).__init__()
        self.sigma = sigma  # 噪声标准差

    def forward(self, x):
        if self.training:  # 只在训练模式下加噪声
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        else:
            return x


# # ----------------------- SEA 第一阶段：多通路→(drug, ori_prot) -----------------------

class SEAttention_MultiPath(nn.Module):
    """
    上下文调制通路注意力 (Context Modulation Pathway Attention)
    
    Q, K, V 仅基于通路特征计算。药物/蛋白拼接成的上下文用于调制 Logits。
    
    输入：
        dim_drug: 药物特征维度
        dim_prot: 蛋白质特征维度
        dim_path: 通路特征维度
        dim_hidden: 隐藏层维度（用于 Q, K, V）
    """

    def __init__(self, dim_drug, dim_prot, dim_path, dim_hidden, dropout=0.1):
        super().__init__()
        self.dim_hidden = dim_hidden
        
        self.map_drug = nn.Linear(dim_drug, dim_hidden, bias=False)
        self.map_prot = nn.Linear(dim_prot, dim_hidden, bias=False)
        
        self.Wq = nn.Linear(dim_path, dim_hidden, bias=False)
        self.Wk = nn.Linear(dim_path, dim_hidden, bias=False)
        self.Wv = nn.Linear(dim_path, dim_hidden, bias=False)
        
        context_input_dim = 2 * dim_hidden + dim_path
        self.W_mod_bias = nn.Linear(context_input_dim, 1, bias=True)
        
        # LayerNorms
        self.norm_q = nn.LayerNorm(dim_hidden)
        self.norm_k = nn.LayerNorm(dim_hidden)
        self.norm_v = nn.LayerNorm(dim_hidden)
        
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(dim_hidden)
        
        self.ffn=nn.Sequential(
            nn.Linear(dim_hidden,dim_hidden*2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dim_hidden*2,dim_hidden)
        )
        self.ly=nn.LayerNorm(dim_hidden)

    def forward(self, x_drug, x_prot,
                dti_edge_index, dti_edge_attr,
                prot_copy2ori, num_protein_ori, ppi_edge_index,dti_edge_path_id):

        device = x_drug.device
        src = dti_edge_index[0]     # drug
        dst = dti_edge_index[1]     # protein copy

        # [与原代码 1. 映射复制蛋白 → 原始蛋白 保持一致]
        is_copy = dst >= num_protein_ori
        ori_part = dst.clone()
        if is_copy.any():
            copy_rel = dst[is_copy] - num_protein_ori
            ori_part[is_copy] = prot_copy2ori[copy_rel]

        pair_keys = torch.stack([src, ori_part], dim=-1)
        uniq, pair_id = torch.unique(pair_keys, return_inverse=True, dim=0)
        U = uniq.size(0)
        pair_drug = uniq[:, 0]
        pair_prot = uniq[:, 1]

        # --------------------------------------------------
        # 2. 上下文特征 (C) 准备：drug' ⊕ prot'
        # --------------------------------------------------
        drug_mapped = self.map_drug(x_drug[src])      # [E, H]
        # 蛋白质使用更新后的特征，但我们加上映射层（如原代码中注释所示），以确保语义一致性
        prot_mapped = self.map_prot(x_prot[dst]) # [E, H]
        
        # 拼接得到上下文 C
        context_vec = torch.cat([
            drug_mapped,
            prot_mapped
        ], dim=-1) # [E, 2H]

        # --------------------------------------------------
        # 3. 计算 Q, K, V（基于通路特征 dti_edge_attr）
        # --------------------------------------------------
        q = self.norm_q(self.Wq(dti_edge_attr)) # [E, H]
        k = self.norm_k(self.Wk(dti_edge_attr)) # [E, H]
        v = self.norm_v(self.Wv(dti_edge_attr)) # [E, H]

        # --------------------------------------------------
        # 4. 组内 Self-Attention (调制版本)
        # --------------------------------------------------
        
        # 原始 Logits (通路自身的相似度)
        # scores[i] 代表 q_i 和 k_i 的相似度（用于 Softmax）
        scores = (q * k).sum(-1) / (self.dim_hidden ** 0.5) # [E]
        
        # 计算上下文偏置 (Modulation Bias)
        # bias 是一个标量，加到原始 score 上。
        # 1. 构造包含通路特征的完整调制输入
        full_modulation_input = torch.cat([context_vec, dti_edge_attr], dim=-1) # [E, 2H + dim_path]

        # 2. 计算上下文偏置 (Modulation Bias)
        # 现在每一条边 (通路实例) 获得的偏置都是独一无二的
        context_bias = self.W_mod_bias(full_modulation_input).squeeze(-1) # [E]
        
        #context_bias = self.W_mod_bias(context_vec).squeeze(-1) # [E]
        
        # 调制后的 Logits
        modulated_scores = scores + context_bias
        
        # 组内 softmax
        alpha = grouped_softmax(modulated_scores, pair_id)

        alpha = self.dropout(alpha)

        contrib = v * alpha.unsqueeze(-1) # [E, H]

        # --------------------------------------------------
        # 5. 聚合到 (drug, ori_prot)
        # --------------------------------------------------
        agg = torch.zeros(U, self.dim_hidden, device=device)
        agg = scatter_add(contrib, pair_id, dim=0, out=agg)

        out=agg+self.ffn(agg)

        out=self.ly(out)

        drug_feats_for_pairs = x_drug[pair_drug]               # [U, dim_drug]
        drug_residual = self.map_drug(drug_feats_for_pairs)    # [U, H]
        prot_feats_for_pairs = x_prot[pair_prot]          # [U, dim_prot]
        prot_residual = self.map_prot(prot_feats_for_pairs)  # [U, H]
        out = out #+ drug_residual+prot_residual
        
        return out, pair_drug, pair_prot
    
from torch_scatter import scatter_max, scatter_add

def grouped_softmax_logits(self, scores, group_ids, num_groups):
    E, L = scores.shape
    group_ids_expand = group_ids.unsqueeze(1).expand(-1, L)  # [E, L]

    # group max
    max_per_group = scatter_max(scores, group_ids_expand, dim=0, dim_size=num_groups)[0]  # [G, L]
    max_expand = max_per_group[group_ids]   # [E, L]

    # exp
    exps = torch.exp(scores - max_expand)

    # group sum
    sum_per_group = scatter_add(exps, group_ids_expand, dim=0, dim_size=num_groups)  # [G, L]
    sum_expand = sum_per_group[group_ids]   # [E, L]

    return exps / (sum_expand + 1e-9)


def grouped_softmax_logits(logits, group_ids, U=None, eps=1e-9):
    """
    对每个 group（group_ids 相同的元素）分别对 logits 做 numerically-stable softmax。
    logits: (E,)
    group_ids: (E,) 取值范围 [0..U-1] 或任意整数
    返回: (E,) 归一化后的权重
    NOTE: 采用简单循环实现以保证兼容性与稳定性（可替换为更快的 segment ops）。
    """
    device = logits.device
    if U is None:
        U = int(group_ids.max().item()) + 1
    out = torch.empty_like(logits)
    uniq = torch.arange(U, device=device)
    # 但为了避免全 U 循环（U 可能比实际少），用实际出现的 groups
    present = torch.unique(group_ids)
    for gid in present:
        mask = (group_ids == gid)
        s = logits[mask]
        s = s - s.max()            # stable
        exp_s = s.exp()
        out[mask] = exp_s / (exp_s.sum() + eps)
    return out



def grouped_softmax(scores, group_idx):
    """按 group_idx 分组 softmax"""
    scores_exp = torch.exp(scores - scores.max())
    denom = torch.zeros_like(scores_exp).index_add_(0, group_idx, scores_exp)
    return scores_exp / (denom[group_idx] + 1e-9)

class SEAttention2_Temp(nn.Module):

    def __init__(self,
                 dim_hidden,
                 num_heads=4,
                 dropout=0.3,
                 mask_file=None,
                 drug2id_file=None,
                 protein2id_file=None,
                 default_mask=1,
                 use_mlp=True,
                 mlp_hidden=None,
                 mlp_dropout=0.3):
        super().__init__()
        assert dim_hidden % num_heads == 0, "dim_hidden 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.dim_head = dim_hidden // num_heads
        self.scale = self.dim_head ** 0.5
        self.default_mask = default_mask

        # ── 修改1：mask_lambda 用 softplus 保证恒正，初始化为 log(e-1)≈0.541
        #    使得 softplus(init) ≈ 1.0，与原始语义一致 ──────────────────────────
        self.mask_lambda_raw = nn.Parameter(torch.tensor(0.5413))

        # ── 掩码加载 ─────────────────────────────────────────────────────────────
        self.mask_dict = {}
        if mask_file and drug2id_file and protein2id_file:
            with open(drug2id_file, 'r', encoding='utf-8') as f:
                self.drug2id = json.load(f)
            with open(protein2id_file, 'r', encoding='utf-8') as f:
                self.protein2id = json.load(f)

            df = pd.read_csv(mask_file)
            matched, missing = 0, 0
            for _, row in df.iterrows():
                drug = row["drug_smi"]
                prot = row["Entry_ID"]
                mask_val = float(row["mask"])
                if drug in self.drug2id and prot in self.protein2id:
                    d_idx = self.drug2id[drug]
                    p_idx = self.protein2id[prot]
                    self.mask_dict[(d_idx, p_idx)] = mask_val
                    matched += 1
                else:
                    missing += 1
            print(f"✅ 掩码加载完成：匹配 {matched} 条，未匹配 {missing} 条，默认值={self.default_mask}")
        else:
            print("⚠️ 未加载掩码文件，将全部使用默认缩放值。")

        # ── 注意力模块 ───────────────────────────────────────────────────────────
        self.Wq = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.Wk = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.Wv = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.Wo = nn.Linear(dim_hidden, dim_hidden, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim_hidden)   # attention 路径的 LayerNorm

        # ── 修改2：MLP 分支增加独立 LayerNorm ────────────────────────────────────
        self.use_mlp = use_mlp
        if self.use_mlp:
            _mlp_hidden  = mlp_hidden  or dim_hidden
            _mlp_dropout = mlp_dropout or dropout
            self.mlp = nn.Sequential(
                nn.Linear(dim_hidden, _mlp_hidden),
                nn.ReLU(),
                nn.Dropout(_mlp_dropout),
                nn.Linear(_mlp_hidden, dim_hidden)
            )
            self.ln2 = nn.LayerNorm(dim_hidden)  # ← 新增，对齐标准 Transformer Block

    # ── 工具方法 ─────────────────────────────────────────────────────────────────
    def get_mask(self, drug_idx, prot_idx):
        return self.mask_dict.get((drug_idx, prot_idx), self.default_mask)

    @property
    def mask_lambda(self):
        """始终为正的可学习缩放因子"""
        return F.softplus(self.mask_lambda_raw)

    # ── 前向传播 ─────────────────────────────────────────────────────────────────
    def forward(self, temp_pair, pair_drug, pair_prot):
        """
        Args:
            temp_pair : Tensor [U, H]  每条边的通路聚合向量
            pair_drug : Tensor [U]     每条边对应的药物 ID
            pair_prot : Tensor [U]     每条边对应的蛋白 ID

        Returns:
            temp_pair_upd : Tensor [U, H]
        """
        device = temp_pair.device
        U, H = temp_pair.size()

        # ── Step 1：查掩码 ───────────────────────────────────────────────────────
        mask_vals = torch.tensor(
            [self.get_mask(d, p) for d, p in zip(pair_drug.tolist(), pair_prot.tolist())],
            dtype=torch.float, device=device
        )  # [U]

        # ── Step 2：组内 softmax 归一化 ──────────────────────────────────────────
        # 同一药物组内归一化，保证不同组的偏置尺度一致
        exp_mask  = torch.exp(mask_vals)
        group_sum = scatter_add(exp_mask, pair_drug, dim=0)         # [num_drugs]
        norm_mask = exp_mask / (group_sum[pair_drug] + 1e-9)        # [U]
        mask_bias = norm_mask.unsqueeze(-1).unsqueeze(-1)            # [U, 1, 1]

        # ── Step 3：多头注意力 QKV ───────────────────────────────────────────────
        q = self.Wq(temp_pair).view(U, self.num_heads, self.dim_head)  # [U, nh, dh]
        k = self.Wk(temp_pair).view(U, self.num_heads, self.dim_head)
        v = self.Wv(temp_pair).view(U, self.num_heads, self.dim_head)

        # ── Step 4：注意力 logits + 掩码偏置 ────────────────────────────────────
        attn_logits = torch.einsum("ihd,jhd->ijh", q, k) / self.scale  # [U, U, nh]

        # 修改1：softplus 保证 mask_lambda > 0，掩码偏置方向正确
        attn_logits = attn_logits + self.mask_lambda * mask_bias

        # 组间隔离：非同药物的 logits 置为 -inf
        group_mask  = (pair_drug.unsqueeze(0) == pair_drug.unsqueeze(1))  # [U, U]
        attn_logits = attn_logits.masked_fill(~group_mask.unsqueeze(-1), float("-inf"))

        # ── Step 5：Softmax + dropout ────────────────────────────────────────────
        attn_weights = F.softmax(attn_logits, dim=1)   # dim=1：对"被查询的边"归一化
        attn_weights = self.dropout(attn_weights)

        # ── Step 6：加权聚合 + 残差 + LN ────────────────────────────────────────
        out = torch.einsum("ijh,jhd->ihd", attn_weights, v).reshape(U, H)
        temp_pair_upd = self.ln(temp_pair + self.Wo(out))

        # ── Step 7：可选 FFN（修改2：加 ln2 + 残差，对齐 Transformer Block）───────
        if self.use_mlp:
            temp_pair_upd = self.ln2(temp_pair_upd + self.mlp(temp_pair_upd))

        return temp_pair_upd

# ----------------------- 顶层编码器：SEA1 → SEA2 → PMA -----------------------
class SEGAHGTEncoder(nn.Module):
    def __init__(self,
                 dim_drug,
                 dim_prot,
                 dim_pathway,
                 dim_hidden,
                 num_heads,
                 metadata,
                 node_num_dict,
                 feat_dim_dict,
                 dropout=0.1,
                 drug2id=None,
                 protein2id=None):
        super().__init__()
        # 分层 SEA
        self.sea1 = SEAttention_MultiPath(
            dim_drug=dim_drug, dim_prot=dim_prot, dim_path=dim_pathway,
            dim_hidden=dim_hidden, dropout=dropout
        )
        self.sea2 = SEAttention2_Temp(
            dim_hidden=dim_hidden,
            num_heads=4,
            dropout=dropout,
            mask_file=r"/home/lkp/cywhome/PITSynergy/PITSynergy/紫杉醇/替换后的文件.csv",
            drug2id_file=r"/home/lkp/cywhome/PITSynergy/PITSynergy/13/drugid小.json",
            protein2id_file=r"/home/lkp/cywhome/PITSynergy/PITSynergy/13/proteinid全.json",
            mlp_dropout=0.3,mlp_hidden=dim_hidden*2
        )

        # 你项目中已有的 SAB/PMA（接口保持一致）
        self.pma = PMAComplete(
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_outputs=1,
            norm_type="LN",
            dropout=0.1,
            use_mlp=True,
            mlp_hidden_size=128,
            #mlp_type="gated_mlp",
            mlp_dropout=0.2,
            #num_mlp_layers=3,
            pre_or_post="pre"
        )

    def forward(self,
                x_dict,
                edge_index_dict=None,
                edge_attr_dict=None,
                # 兼容旧签名，但我们主要用下列三个输入：
                dti_edge_index=None,  # hetero_data['drug','interacts','protein'].edge_index  (指向复制蛋白)
                dti_edge_attr=None,  # hetero_data['drug','interacts','protein'].edge_attr   (通路embedding)
                prot_copy2ori=None,  # hetero_data['protein'].prot_copy2ori  (len = 复制蛋白数)
                ppi_edge_index=None,
                dti_edge_path_id=None
                ):
        """
        返回：
          drug_structure_repr: x_dict['drug']（保持结构通道不变）
          drug_semantic_repr:  [Nd, H]
        """
        x_drug = x_dict['drug']  # [Nd, Dd]
        x_prot = x_dict['protein']  # [Np_total, Dp]
        Nd = x_drug.size(0)
        Np_total = x_prot.size(0)
        Nc = 0 if prot_copy2ori is None else prot_copy2ori.numel()
        num_protein_ori = Np_total - Nc

        # 1) SEA1：按 (drug, 原始蛋白) 组内聚合多通路
        temp_pair, pair_drug, pair_prot= self.sea1(
            x_drug=x_drug,
            x_prot=x_prot,
            dti_edge_index=dti_edge_index,
            dti_edge_attr=dti_edge_attr,
            prot_copy2ori=prot_copy2ori if Nc > 0 else torch.empty(0, dtype=torch.long, device=x_prot.device),
            num_protein_ori=num_protein_ori,
            ppi_edge_index=ppi_edge_index,
            dti_edge_path_id=dti_edge_path_id
        )  # [U,H], [U], [U]

        temp_pair = self.sea2(temp_pair=temp_pair, pair_drug=pair_drug, pair_prot=pair_prot)

        drug_semantic_repr = self.pma(
            X_edge_features=temp_pair,
            edge_index=dti_edge_index,  # <== 加上这个
            batch_mapping=pair_drug
        ).squeeze(1)  # [Nd,H]
        return x_drug, drug_semantic_repr
