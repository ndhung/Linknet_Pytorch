import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data


class CamvidLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, img_size=None):
        self.root = root
        self.split = split
        self.img_size = [576, 768]
        # self.img_size = [352, 480]
        self.is_transform = is_transform
        self.mean = np.array([103.939, 116.779, 123.68])
        # self.mean = np.array([122.67892, 116.66877, 104.00699])
        self.n_classes = 12
        self.files = collections.defaultdict(list)

        for split in ["train", "test", "val"]:
            file_list = [file for file in os.listdir(root + '/' + split) if file.endswith('.png')]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + '/' + self.split + '/' + img_name
        lbl_path = self.root + '/' + self.split + 'annot/' + img_name

        img = m.imread(img_path)
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = np.array(img, dtype=np.uint8)
    
        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        lbl = lbl.astype(float)
        lbl = m.imresize(
            lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        Road = [255, 69, 0]
        Pavement = [128, 64, 128]
        Tree = [60, 40, 222]
        SignSymbol = [128, 128, 0]
        Fence = [192, 128, 128]
        Car = [64, 64, 128]
        Pedestrian = [64, 0, 128]
        Bicyclist = [64, 64, 0]
        Unlabeled = [0, 128, 192]

        label_colours = np.array([Sky, Building, Pole, Road,
                                  Pavement, Tree, SignSymbol, Fence, Car,
                                  Pedestrian, Bicyclist, Unlabeled])
        r = np.zeros_like(temp)
        g = np.zeros_like(temp)
        b = np.zeros_like(temp)
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = b
        rgb[:, :, 1] = g
        rgb[:, :, 2] = r
        rgb = np.array(rgb, dtype=np.uint8)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    local_path = '/home/vietdoan/Workingspace/Enet/camvid'
    dst = CamvidLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 3:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            plt.imshow(dst.decode_segmap(labels.numpy()[0]))
            plt.show()