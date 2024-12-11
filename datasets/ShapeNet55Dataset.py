import os
import torch
import numpy as np
import torch.utils.data as data
from datasets.io import IO

class ShapeNet(data.Dataset):
    def __init__(self, subset, whole):
        self.data_root = "/root/wish/Point-NN-main/data/ShapeNet55-34/ShapeNet-55"
        self.pc_path = "/root/wish/Point-NN-main/data/ShapeNet55-34/shapenet_pc"
        self.subset = subset
        self.npoints = 1024

        self.data_list_file = os.path.join(
            self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = 1024
        self.whole = whole

        # print_log(
        #     f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        # print_log(
        #     f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            # print_log(
            #     f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        self.taxonomy_ids = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
            if taxonomy_id not in self.taxonomy_ids:
                self.taxonomy_ids.append(taxonomy_id)

        self.taxonomy_ids.sort()
        # print_log(
        #     f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(
            self.pc_path, sample['file_path'])).astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        x = torch.cat((data, data[:, 1:2] - data[:, 1:2].min()), dim=1)
        # return data, data, sample['taxonomy_id']
        return x, data, self.taxonomy_ids.index(sample['taxonomy_id'])


    def __len__(self):
        return len(self.file_list)
