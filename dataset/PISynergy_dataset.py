import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from .base_InMemory_dataset import BaseInMemoryDataset

# 处理细胞系和药物数据
class MyInMemoryDataset(BaseInMemoryDataset):
    def __init__(self,
                 data_root,  # 数据的路径
                 data_items, # 训练集数据集的.npy文件
                 celllines_data, # 细胞系的多组学数据
                 drugs_data,     # 药物的子结构数据
                 transform=None,
                 pre_transform=None,
                 args = None):

        super(MyInMemoryDataset, self).__init__(root=data_root, transform=transform, pre_transform=pre_transform)


        if args.celldataset == 1:
            self.name = osp.basename(data_items).split('items')[0]+'18498g'
        elif args.celldataset == 2:
            self.name = osp.basename(data_items).split('items')[0]+'4079g'
        elif args.celldataset == 3:
            self.name = osp.basename(data_items).split('items')[0]+'963g'

        self.name = self.name+'_TransDrug_norm'

        self.args = args
        self.data_items = np.load(data_items, allow_pickle=True)
        self.celllines = np.load(celllines_data, allow_pickle=True).item()
        self.drugs = np.load(drugs_data, allow_pickle=True).item()

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        data_list = []
        data_len = len(self.data_items)

        for i in tqdm(range(data_len)):

            drugA, drugB, c1, label = self.data_items[i]
            cell_features = self.celllines[c1]

            drugA_features = self.drugs[drugA]
            drugB_features = self.drugs[drugB]

            cell_drug_data = Data()

            cell_drug_data.drugA = torch.Tensor(np.array([drugA_features])).to(dtype=torch.float16)
            cell_drug_data.drugB = torch.Tensor(np.array([drugB_features])).to(dtype=torch.float16)

            cell_drug_data.x_cell = torch.as_tensor(cell_features).to(dtype=torch.float16)
            cell_drug_data.y = torch.Tensor([float(label)]).to(dtype=torch.float16)

            data_list.append(cell_drug_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('data construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print('Dataset construction done.')
