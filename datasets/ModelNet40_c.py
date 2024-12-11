import os
from collections import defaultdict

import h5py
from torch.utils.data import Dataset
#from augmentation.PointWOLF.PointWOLF import PointWOLF

DATA_DIR = "/root/wish/Point-NN-main/data/modelnet_c"


def load_h5(h5_name):
    #print(h5_name)

    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data, label


class ModelNetC(Dataset):
    def __init__(self, args, split):
        h5_path = os.path.join(DATA_DIR, split + '.h5')
        self.data, self.label = load_h5(h5_path)
        #if args.use_wolfmix:
        #    self.PointWOLF = PointWOLF(args)

        # print(self.data.shape, self.label.shape)
        # print(h5_path)
    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        # label_name = self.modelnet40_label[int(label)]

        # print(label_name)
        return pointcloud, pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ObjaverseC(Dataset):
    def __init__(self, args, split):
        h5_path = os.path.join(DATA_DIR, 'objaverse_' + split + '.h5')
        self.data, self.label = load_h5(h5_path)
        # if args.use_wolfmix:
        #    self.PointWOLF = PointWOLF(args)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        # label_name = self.modelnet40_label[int(label)]

        # print(label_name)
        return pointcloud, pointcloud, label

    def __len__(self):
        return self.data.shape[0]

        
if __name__ == '__main__':
   # train = ScanObjectNN(None, 'train')
   # test = ScanObjectNN(None, 'test')
   # pts,label = train.__getitem__(0)
   # print(pts.shape, label.shape)
   pass

