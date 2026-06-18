from rdkit import Chem
import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn as nn



class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''
    def __inc__(self, key, value, *args, **kwargs):
    # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
    # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement()!=0 else 0
        return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"



class EdgeFeatureMapping(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(EdgeFeatureMapping, self).__init__()
        # 定义线性层，将输入维度映射到64维
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, edge_features):
        # 通过线性层进行映射
        return self.linear(edge_features)


# ========== 工具函数：one-hot 编码 ==========

'''def one_of_k_encoding(value, num_classes):
    encoding = np.zeros(num_classes)
    encoding[value] = 1
    return torch.tensor(encoding, dtype=torch.float)


def one_of_k_encoding_unk(value, num_classes):
    if value >= num_classes:
        value = num_classes - 1  # 设置为未知类别（unkown）
    return one_of_k_encoding(value, num_classes)

def bond_features(bond):
    return torch.tensor([
        int(bond.GetBondTypeAsDouble()),      # 单双三键
        int(bond.GetIsConjugated()),          # 是否共轭
        int(bond.IsInRing())                  # 是否在环中
    ], dtype=torch.float)



# ========== 工具函数：原子编码 ==========

def atom_features(atom, use_chirality=False):
    # 1. 原子符号的 one-hot 编码
    atom_symbol = atom.GetSymbol()
    symbol_to_idx = {
        'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10,
        'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'K': 17, 'Ar': 18, 'Ca': 19, 'Sc': 20,
        'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ga': 30
    }
    symbol_idx = symbol_to_idx.get(atom_symbol, 31)  # 默认为未知元素
    symbol_encoding = one_of_k_encoding(symbol_idx, 31)

    # 2. 原子度数的 one-hot 编码
    degree = atom.GetDegree()  # 0-10
    degree_encoding = one_of_k_encoding(degree, 11)

    # 3. 原子隐性价的 one-hot 编码
    valence = atom.GetImplicitValence()  # 0-6
    valence_encoding = one_of_k_encoding_unk(valence, 7)

    # 4. 电荷
    charge = atom.GetFormalCharge()
    charge_encoding = torch.tensor([charge], dtype=torch.float)

    # 5. 自由电子数
    radical_electrons = atom.GetNumRadicalElectrons()
    radical_encoding = torch.tensor([radical_electrons], dtype=torch.float)

    # 6. 杂化类型的 one-hot 编码
    hybridization = atom.GetHybridization()
    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4
    }
    hybridization_encoding = one_of_k_encoding_unk(hybridization_map.get(hybridization, 4), 5)

    # 7. 芳香性
    aromatic = int(atom.GetIsAromatic())
    aromatic_encoding = torch.tensor([aromatic], dtype=torch.float)

    # 8. 氢原子的数量的 one-hot 编码
    num_hydrogens = atom.GetTotalNumHs()  # 0-4
    hydrogen_encoding = one_of_k_encoding_unk(num_hydrogens, 5)

    # 9. 手性特征（如果需要的话）
    chirality_encoding = torch.tensor([0, 0], dtype=torch.float)  # 默认不使用手性
    if use_chirality and atom.HasProp('_CIPCode'):
        cip_code = atom.GetProp('_CIPCode')
        if cip_code == 'R':
            chirality_encoding = torch.tensor([1, 0], dtype=torch.float)
        elif cip_code == 'S':
            chirality_encoding = torch.tensor([0, 1], dtype=torch.float)

    # 合并所有特征
    atom_feature_vector = torch.cat([
        symbol_encoding, degree_encoding, valence_encoding,
        charge_encoding, radical_encoding, hybridization_encoding,
        aromatic_encoding, hydrogen_encoding, chirality_encoding
    ], dim=0)

    return atom_feature_vector


# ========== 主函数 ==========

def mol_to_pyg_data(mol, use_chirality=False):
    mol = Chem.AddHs(mol)  # 添加隐式氢
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # 生成原子特征 x，每个原子有64维特征
    x = torch.stack([atom_features(atom, use_chirality) for atom in mol.GetAtoms()])

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        f = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [f, f]

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    n_edges = edge_index.size(1)
    src = edge_index[0]
    tgt = edge_index[1]
    edge_ids = {(src[i].item(), tgt[i].item()): i for i in range(n_edges)}

    line_graph_edge_index = []

    for i in range(n_edges):
        tgt_i = tgt[i].item()
        for j in range(n_edges):
            if i == j:
                continue
            src_j = src[j].item()
            if tgt_i == src_j:
                line_graph_edge_index.append([i, j])

    if line_graph_edge_index:
        line_graph_edge_index = torch.tensor(line_graph_edge_index).t().contiguous()
    else:
        line_graph_edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_index_batch = torch.zeros(edge_index.size(1), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        line_graph_edge_index=line_graph_edge_index,
        edge_index_batch=edge_index_batch
    )

    return data'''



# ========== 工具函数：one-hot 编码 ==========

'''def one_of_k_encoding(value, num_classes):
    encoding = np.zeros(num_classes)
    encoding[value] = 1
    return torch.tensor(encoding, dtype=torch.float)'''
def one_of_k_encoding(value, size):
    encoding = torch.zeros(size, dtype=torch.float)
    if value < size:
        encoding[value] = 1
    else:
        # 处理非法值，可以选择抛出异常或将其设置为默认值
        print(f"Warning: value {value} is out of range, setting to 0")
    return encoding



def one_of_k_encoding_unk(value, num_classes):
    if value >= num_classes:
        value = num_classes - 1  # 设置为未知类别（unkown）
    return one_of_k_encoding(value, num_classes)

'''def bond_features(bond):
    return torch.tensor([
        int(bond.GetBondTypeAsDouble()),      # 单双三键
        int(bond.GetIsConjugated()),          # 是否共轭
        int(bond.IsInRing())                  # 是否在环中
    ], dtype=torch.float)'''
def bond_features(bond):

    '''
    Get bond features
    '''
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()



# ========== 工具函数：原子编码 ==========

def atom_features(atom, use_chirality=False):
    # 1. 原子符号的 one-hot 编码
    atom_symbol = atom.GetSymbol()
    symbol_to_idx = {
        'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 'Na': 10,
        'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'K': 17, 'Ar': 18, 'Ca': 19, 'Sc': 20,
        'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ga': 30
    }
    symbol_idx = symbol_to_idx.get(atom_symbol, 31)  # 默认为未知元素
    symbol_encoding = one_of_k_encoding(symbol_idx, 31)

    # 2. 原子度数的 one-hot 编码
    degree = atom.GetDegree()  # 0-10
    degree_encoding = one_of_k_encoding(degree, 11)

    # 3. 原子隐性价的 one-hot 编码
    valence = atom.GetImplicitValence()  # 0-6
    valence_encoding = one_of_k_encoding_unk(valence, 7)

    # 4. 电荷
    charge = atom.GetFormalCharge()
    charge_encoding = torch.tensor([charge], dtype=torch.float)

    # 5. 自由电子数
    radical_electrons = atom.GetNumRadicalElectrons()
    radical_encoding = torch.tensor([radical_electrons], dtype=torch.float)

    # 6. 杂化类型的 one-hot 编码
    hybridization = atom.GetHybridization()
    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4
    }
    hybridization_encoding = one_of_k_encoding_unk(hybridization_map.get(hybridization, 4), 5)

    # 7. 芳香性
    aromatic = int(atom.GetIsAromatic())
    aromatic_encoding = torch.tensor([aromatic], dtype=torch.float)

    # 8. 氢原子的数量的 one-hot 编码
    num_hydrogens = atom.GetTotalNumHs()  # 0-4
    hydrogen_encoding = one_of_k_encoding_unk(num_hydrogens, 5)

    # 9. 手性特征（如果需要的话）
    chirality_encoding = torch.tensor([0, 0], dtype=torch.float)  # 默认不使用手性
    if use_chirality and atom.HasProp('_CIPCode'):
        cip_code = atom.GetProp('_CIPCode')
        if cip_code == 'R':
            chirality_encoding = torch.tensor([1, 0], dtype=torch.float)
        elif cip_code == 'S':
            chirality_encoding = torch.tensor([0, 1], dtype=torch.float)

    # 合并所有特征
    atom_feature_vector = torch.cat([
        symbol_encoding, degree_encoding, valence_encoding,
        charge_encoding, radical_encoding, hybridization_encoding,
        aromatic_encoding, hydrogen_encoding, chirality_encoding
    ], dim=0)

    return atom_feature_vector


# ========== 主函数 ==========
'''def mol_to_pyg_data(mol, use_chirality=False):
    mol = Chem.AddHs(mol)  # 添加隐式氢
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # 生成原子特征 x，每个原子有64维特征
    x = torch.stack([atom_features(atom, use_chirality) for atom in mol.GetAtoms()])

    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        f = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [f, f]

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    # 关键：确保 edge_attr 转为 Float 类型，并通过线性层映射到 64 维
    edge_attr = edge_attr.float()  # 修改为 Float 类型
    edge_feature_mapping = EdgeFeatureMapping(input_dim=edge_attr.size(1), output_dim=64)  # 假设原始维度为edge_attr.size(1)
    edge_attr = edge_feature_mapping(edge_attr)  # 映射到 64 维

    n_edges = edge_index.size(1)
    src = edge_index[0]
    tgt = edge_index[1]
    edge_ids = {(src[i].item(), tgt[i].item()): i for i in range(n_edges)}

    line_graph_edge_index = []

    for i in range(n_edges):
        tgt_i = tgt[i].item()
        for j in range(n_edges):
            if i == j:
                continue
            src_j = src[j].item()
            if tgt_i == src_j:
                line_graph_edge_index.append([i, j])

    if line_graph_edge_index:
        line_graph_edge_index = torch.tensor(line_graph_edge_index).t().contiguous()
    else:
        line_graph_edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_index_batch = torch.zeros(edge_index.size(1), dtype=torch.long)

    data = CustomData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        line_graph_edge_index=line_graph_edge_index
    )

    return data

'''
def mol_to_pyg_data(mol, use_chirality=False):
    mol = Chem.AddHs(mol)  # 添加隐式氢
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()

    # 原子特征
    atom_features_list = [atom_features(atom, use_chirality) for atom in mol.GetAtoms()]
    x_atoms = torch.stack(atom_features_list)

    # 边特征
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        f = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [f, f]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.stack(edge_attr).float()  # [num_edges, bond_feature_dim]

    # ============== 新加：构建 line graph ==============
    n_edges = edge_index.size(1)
    src = edge_index[0]
    tgt = edge_index[1]

    line_graph_edge_index = []
    for i in range(n_edges):
        tgt_i = tgt[i].item()
        for j in range(n_edges):
            if i == j:
                continue
            src_j = src[j].item()
            if tgt_i == src_j:
                line_graph_edge_index.append([i, j])

    if line_graph_edge_index:
        line_graph_edge_index = torch.tensor(line_graph_edge_index, dtype=torch.long).t().contiguous()
    else:
        line_graph_edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_index_batch = torch.zeros(edge_index.size(1), dtype=torch.long)

    data = CustomData(
        x=x_atoms,  # 原子特征
        edge_index=edge_index,
        edge_attr=edge_attr,
        line_graph_edge_index=line_graph_edge_index
    )

    return data



