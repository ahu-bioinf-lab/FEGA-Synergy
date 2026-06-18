import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

def build_drug_target_graph(csv_path):
    df = pd.read_csv(csv_path)

    # 解析每行的 Entry_ID
    df['Entry_ID'] = df['Entry_ID'].apply(lambda x: [s.strip() for s in str(x).split(',')])

    # 收集所有 unique 药物和蛋白
    all_drugs = df['PubChem_CID'].astype(str).unique().tolist()
    all_targets = sorted(set(tgt for lst in df['Entry_ID'] for tgt in lst))

    drug_encoder = LabelEncoder().fit(all_drugs)
    target_encoder = LabelEncoder().fit(all_targets)

    num_drugs = len(all_drugs)
    num_targets = len(all_targets)

    # 构建边
    drug_nodes = []
    target_nodes = []

    for _, row in df.iterrows():
        drug_id = str(row['PubChem_CID'])
        for target in row['Entry_ID']:
            drug_nodes.append(drug_encoder.transform([drug_id])[0])
            target_nodes.append(target_encoder.transform([target])[0])

    drug_nodes = torch.tensor(drug_nodes, dtype=torch.long)
    target_nodes = torch.tensor(target_nodes, dtype=torch.long)

    data = HeteroData()

    # 用 one-hot 作为初始特征
    data['drug'].x = torch.eye(num_drugs)
    data['protein'].x = torch.eye(num_targets)

    # 添加双向边
    data['drug', 'targets', 'protein'].edge_index = torch.stack([drug_nodes, target_nodes], dim=0)
    data['protein', 'targeted_by', 'drug'].edge_index = torch.stack([target_nodes, drug_nodes], dim=0)

    # 可选：保存编码器映射关系
    data['drug'].pubchem_id = all_drugs
    data['protein'].entry_id = all_targets

    return data
