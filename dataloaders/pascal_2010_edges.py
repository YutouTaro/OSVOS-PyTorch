from __future__ import division

import os
import numpy as np
import cv2
from PIL import Image

from dataloaders.helpers import *
from torch.utils.data import Dataset


class PASCAL2010(Dataset):
    """PASCAL 2010 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='/content/drive/My Drive/dataset/PASCAL2010',
                 transform=None,
                 meanval=(116.83839, 111.93566, 103.42352),
                 # seq_name=None
                 ):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        # self.seq_name = seq_name

        if self.train:
            fname = 'train'
        else:
            fname = 'test'

        # if self.seq_name is None:
        #
        #     # Initialize the original DAVIS splits for training the parent network
        #     with open(os.path.join(db_root_dir, fname + '.txt')) as f:
        #         seqs = f.readlines()
        #         img_list = []
        #         labels = []
        #         for seq in seqs:
        #             images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))
        #             images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
        #             img_list.extend(images_path)
        #             lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
        #             lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
        #             labels.extend(lab_path)
        # else:
        #
        #     # Initialize the per sequence images for online training
        #     names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', str(seq_name))))
        #     img_list = list(map(lambda x: os.path.join('JPEGImages/480p/', str(seq_name), x), names_img))
        #     name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', str(seq_name))))
        #     labels = [os.path.join('Annotations/480p/', str(seq_name), name_label[0])]
        #     labels.extend([None]*(len(names_img)-1))
        #     if self.train:
        #         img_list = [img_list[0]]
        #         labels = [labels[0]]
        images = np.sort(os.listdir(os.path.join(db_root_dir, 'VOC2010', fname)))
        img_list = list(map(lambda x: os.path.join('VOC2010', fname, x), images))
        labs = np.sort(os.listdir(os.path.join(db_root_dir, 'edges', fname)))
        labels = list(map(lambda x: os.path.join('edges', fname, x), labs))



        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        # if self.seq_name is not None:
        #     fname = os.path.join(self.seq_name, "%05d" % idx)
        #     sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            # img = imresize(img, self.inputRes)
            img = np.array(Image.fromarray(img).resize(self.inputRes))
            if self.labels[idx] is not None:
                # label = imresize(label, self.inputRes, interp='nearest')
                label = np.array(Image.fromarray(label).resize(self.inputRes, resample=Image.NEAREST))

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
                gt = np.array(label, dtype=np.float32)
                gt = gt/np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    dataset = PASCAL2010(db_root_dir='/content/drive/My Drive/datasets/PASCAL2010',
                        train=False, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
        if i == 10:
            break

    plt.show(block=True)
