import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from DMPNN import *
from mol_to_linegraph import *

# added
def smile2pyg(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")

    pyg_data = mol_to_pyg_data(mol)  # 返回含 line graph 的 PyG Data 对象
    pyg_data = pyg_data.to('cuda')
    pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)
    return pyg_data

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


# 处理药物drugs.csv -> drugs.npy
# def drugConvert(filePath, datasetName, drug_sub_dict):
#
#     # 读取drugs.csv文件
#     drug = pd.read_csv(filePath + datasetName + "/drugs.csv", index_col=0)
#     for key, value in drug_sub_dict.items():
#         temp_value = [[elem for elem in arr] for arr in value]
#         drug_sub_dict[key] = temp_value;
#
#     # 使用 apply 函数将替换函数应用到 drug_smi 列
#     drug['drug_smi'] = drug['drug_smi'].apply(lambda x: drug_sub_dict.get(x, x))
#
#     # 保存数据到 .pt 文件
#     torch.save(drug, filePath + datasetName + "/drugs.pt")
#     print("drug converted")

'''def drugConvert(filePath, datasetName, hidden_dim=64, n_iter=3, model_path=None):
    drug_df = pd.read_csv(filePath + datasetName + "/drugs.csv", index_col=0)
    smiles_list = drug_df['drug_smi'].tolist()

    # 初始化编码器
    model = MPNN_Block(hidden_dim, n_iter)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to('cuda')

    drug_features = []
    for smi in tqdm(smiles_list, desc="Encoding drugs"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        pyg_data = mol_to_pyg_data(mol)  # 必须返回含 line graph 信息的 PyG Data 对象
        pyg_data = pyg_data.to('cuda')
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)

        with torch.no_grad():
            _, drug_embedding = model(pyg_data)  # 只用到 readout 的图级嵌入
            drug_features.append(drug_embedding.squeeze(0).cpu())

    # 用 'drug_feature' 列替代 'drug_smi' 列的位置
    drug_df['drug_smi'] = drug_features  # 直接替换 'drug_smi' 列

    # 保存新的 DataFrame
    # 保存为 Tensor（推荐）
    drug_tensor = torch.stack(drug_features, dim=0)  # [num_drugs, feature_dim]
    torch.save(drug_tensor, filePath + datasetName + "/drugs1.pt")
    print("drug converted by DMPNN and saved as tensor.")'''


def drugConvert(filePath, datasetName, hidden_dim=64, n_iter=3, model_path=None):
    # 读取 SMILES
    drug_df = pd.read_csv(filePath + datasetName + "/drugs.csv", index_col=0)
    smiles_list = drug_df['drug_smi'].tolist()

    # 初始化编码器
    model = MPNN_Block(hidden_dim, n_iter)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to('cuda')

    drug_features = []
    for smi in tqdm(smiles_list, desc="Encoding drugs"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        pyg_data = mol_to_pyg_data(mol)  # 返回含 line graph 的 PyG Data 对象
        pyg_data = pyg_data.to('cuda')
        pyg_data.batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=pyg_data.x.device)

        with torch.no_grad():
            _, drug_embedding = model(pyg_data)  # 图级嵌入
            drug_features.append(drug_embedding.squeeze(0).cpu())

    # 构建两列的 DataFrame：drug_id, drug_feature
    drug_ids = list(range(len(drug_features)))  # 或用 drug_df.index.tolist()
    df = pd.DataFrame({
        'drug_smi': drug_ids,
        'drug_feature': drug_features
    })

    # 保存为 .pt 文件（DataFrame 格式）
    torch.save(df, filePath + datasetName + "/drugs2.pt")
    print("Drug features saved as DataFrame with two columns (drug_id, drug_feature).")


# 处理细胞系clines.csv -> clines.npy
def clineConvert(filePath, datasetName, cell_muti_dict):
    # 读取drugs.csv文件
    cline = pd.read_csv(filePath + datasetName + "/clines.csv", index_col=0)
    for key, value in cell_muti_dict.items():
        temp_value = [[elem for elem in arr] for arr in value]
        # temp_value = torch.as_tensor(value)
        cell_muti_dict[key] = temp_value

    # 这里的cell_id 就是 DepMap_ID
    cline['cell_id'] = cline['cell_id'].apply(lambda x: cell_muti_dict.get(x, x))

    # 保存数据到 .pt 文件
    torch.save(cline, filePath + datasetName + "/clines.pt")
    # np.save(filePath + datasetName + "/clines.npy", cline.to_numpy())
    print("cline converted")

# 处理协同数据 synergy.csv -> synergy_id.csv
def synergyConvert(filePath, datasetName):
    # 读取drugs.csv文件
    drug = pd.read_csv(filePath + datasetName + "/drugs.csv")
    # 读取drugs.csv文件
    cline = pd.read_csv(filePath + datasetName + "/clines.csv")
    # 读取synergy.csv 文件
    synergy = pd.read_csv(filePath+ datasetName+"/synergy.csv")
    #  将第三个DataFrame中的 'cell_id' 替换为对应的索引值
    synergy['cell_id'] = synergy['cell_id'].map(cline.set_index('cell_id')['index'])
    # 或者 df3['cell_id'] = df3['cell_id'].map(df1.set_index('cell_id')['index1'].to_dict())

    # 将第三个DataFrame中的 'druga_smi' 和 'drugb_smi' 替换为对应的索引值
    # synergy['druga_smi'] = synergy['druga_smi'].map(drug.set_index('drug_smi')['index'])
    # synergy['drugb_smi'] = synergy['drugb_smi'].map(drug.set_index('drug_smi')['index'])

    # 4176
    pyg_a_list = []
    pyg_b_list = []
    li = list(enumerate(zip(synergy['druga_smi'].tolist(), synergy['drugb_smi'].tolist())))
    print(f'len(li): {len(li)}')
    for (i, smis) in li:
        smia, smib = smis
        pyg_a = smile2pyg(smia)
        pyg_b = smile2pyg(smib)
        pyg_a_list.append(pyg_a)
        pyg_b_list.append(pyg_b)
        print(f'i: {i}')
    synergy['druga_smi'] = pyg_a_list
    synergy['drugb_smi'] = pyg_b_list

    # synergy.to_csv(filePath + datasetName + "/synergy_id.csv",index=False)
    synergy.to_csv(filePath + datasetName + "/synergy_id_smile2.csv",index=False)
    # 按照drug 和 cline 转换为 id
    print("synergy converted")

# 处理协同数据 synergy.csv -> synergy_id.csv
def synergyConvert_id2tensor(synergy, drugs, clines):
    drugs_dict = drugs['drug_smi'].to_dict()
    clines_dict = clines['cell_id'].to_dict()
    # 将第三个DataFrame中的 'druga_smi' 和 'drugb_smi' 替换为对应的值
    # modified
    # synergy['druga_smi'] = synergy['druga_smi'].map(drugs_dict)
    # synergy['drugb_smi'] = synergy['drugb_smi'].map(drugs_dict)
    synergy['cell_id'] = synergy['cell_id'].map(clines_dict)  # [4079, 4079, 6]

    return synergy

# 处理协同数据 label -> 01
def synergyConvert_label2zeroone(filePath, datasetName):

    # 读取 synergy_id.csv 文件
    synergy = pd.read_csv(filePath+ datasetName+ "/synergy_id.csv")
    threshold = 0
    for i in range(synergy.shape[0]):
         synergy.loc[i,'synergy'] = 1 if synergy.loc[i,'synergy'] >= threshold else 0

    synergy.to_csv(filePath + datasetName + "/synergy_c.csv", index=False)

if __name__ == "__main__" :
    # 读取细胞系和药物的映射文件，是两个字典。
    cell_muti_dict = np.load("./data/cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy",allow_pickle=True).item()
    drug_sub_dict = np.load("./data/drug_data/drugSmile_drugSubEmbed_2644.npy",allow_pickle=True).item()

    filePath = "./data/independent dataset/"

    #datasetName = {"indep0-oneil", "indep1-almanac", "indep2-OncologyScreen", "indep3-DrugCombDB"}
    datasetName = {"indep2-OncologyScreen"}

    for dataset in datasetName:
        # drugConvert(filePath, dataset, drug_sub_dict)
        # clineConvert(filePath, dataset, cell_muti_dict)
        synergyConvert(filePath, dataset)
        # synergyConvert_label2zeroone(filePath, dataset)+
        # drugConvert(filePath, dataset, hidden_dim=64, n_iter=3, model_path=None)

'''
步骤为
1. 首先需要不同数据集的药物smile串, 细胞系id, 以及药物协同数据 [druga_smile,drugb_smile,cline_id,synergy_score],一共三个文件。
2. 得到之后，将药物数据按照./data/drug_data/drugSmile_drugSubEmbed_2644.npy将smile串转换为子结构。
3. 将细胞系的特征按照./data/cell_data/4079/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy 将ID转换为多组学特征。
4. 将协同数据按照smiles和Cline ID转换为序号。[druga_smile,drugb_smile,cline_id,synergy_score] -> [druga_index,drugb_index,cline_index,synergy_score]
'''