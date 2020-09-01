'''
data generator for the global local net

v0: for resnet34 only
v1: for global local with only local path, prepare the data for the input['local']
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import pandas as pd
import random
import os
import math
# from skimage import io, transform
import numpy as np
import cv2
from time import time
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
import imgaug as ia
from imgaug import augmenters as iaa

from sampler import BalancedBatchSampler

plt.ion()

class dataconfig(object):
    def __init__(self, dataset = 'defaut',subset = '0', **kwargs):
        self.dataset = dataset
        self.dir = r'C:\Users\Xing\Projects\TB2020\code\ecg_classification_torch\code'
        self.csv = 'TB_label_0426.csv'
        self.subset = subset
        self.csv_file = os.path.join(self.dir,self.csv)

class batch_sampler():
    def __init__(self, batch_size, class_list):
        self.batch_size = batch_size
        self.class_list = class_list
        self.unique_value = np.unique(class_list)
        self.iter_list = []
        self.len_list = []
        for v in self.unique_value:
            indexes = np.where(self.class_list == v)[0]
            self.iter_list.append(self.shuffle_iterator(indexes))
            self.len_list.append(len(indexes))
        self.len = len(class_list) // batch_size
        # print('self.len: ', self.len)

    def __iter__(self):
        index_list = []
        for _ in range(self.len):
            for index in range(self.batch_size):
                index_list.append(next(self.iter_list[index % len(self.unique_value)]))
            np.random.shuffle(index_list)
            yield index_list
            index_list = []

    def __len__(self):
        return self.len

    @staticmethod
    def shuffle_iterator(iterator):
        # iterator should have limited size
        index = list(iterator)
        total_size = len(index)
        i = 0
        random.shuffle(index)
        while True:
            yield index[i]
            i += 1
            if i >= total_size:
                i = 0
                random.shuffle(index)


class DataGenerator(Dataset):
    def __init__(self, config=None,transform = None,type='0.simplist'):
        self.config = config
        self.imgsz = 256
        self.conlen = 32
        self.debug = False
        self.df = self.parse_csv(self.config.csv_file, self.config.subset)
        self.df.reset_index(drop=True, inplace=True)
        self.transform = transform
        self.type = type

    def __len__(self):
        # if self.config.subset == '0':
        #     print('len = {}'.format(2*len(self.df)))
        #     return 2*len(self.df)
        print('len = {}'.format(len(self.df)))
        return len(self.df)

    def img_augmentation(self,img,seq_det):



        img = img.transpose(2,0,1).astype(np.float32)

        for i in range(len(img)):
            img[i,:,:] = seq_det.augment_images(img[i,:,:])

        img = img.transpose(1,2,0).astype(np.float64)

        return img


    def img_generator(self,index,data,con_len,ss=1):
        # 0.simplist just pick the center 48 slices from data and msk and concat them together

        img = np.zeros([self.imgsz, self.imgsz, con_len])
        img_l = np.zeros([self.imgsz, self.imgsz, con_len])
        img_s = np.zeros([self.imgsz, self.imgsz, con_len])
        img_b = np.zeros([self.imgsz, self.imgsz, con_len])
        msk = np.zeros([self.imgsz, self.imgsz, con_len])
        msk1 = np.zeros([self.imgsz, self.imgsz, con_len])

        if self.type == '0.simplist':
            img_len = self.df.loc[index,'len']
            center_i = img_len // 2


            if con_len <= img_len:
                start_i = center_i - con_len//2
                end_i = start_i+con_len
                img = data['data'][:, :, start_i:end_i]
                msk = data['mask'][:, :, start_i:end_i]
            else:
                start_i = con_len//2 - img_len//2
                end_i = start_i+img_len
                img[:,:,start_i:end_i] = data['data']
                msk[:,:,start_i:end_i] = data['mask']

        elif self.type == '1: slice_sampled':

            img_len = self.df.loc[index, 'len']
            center_i = img_len // 2

            ds = self.df.loc[index,'ds']
            if ds == 1:
                ds = ss*ds
            else:
                ds = ss*ds//2

            if ds*con_len <= img_len:
                start_i = center_i - ds*con_len // 2
                end_i = start_i + ds*con_len
                img = data['data'][:, :, start_i:end_i:ds]
                msk = data['mask'][:, :, start_i:end_i:ds]
                #msk1 = data['mask1'][:, :, start_i:end_i:ds]
            else:
                start_i = con_len // 2 - img_len // 2 //ds
                end_i = start_i + img_len//ds
                img[:, :, start_i:end_i] = data['data'][:,:,0:img_len:ds]
                msk[:, :, start_i:end_i] = data['mask'][:,:,0:img_len:ds]
                #msk1 = data['mask1'][:, :, start_i:end_i:ds]
                print(index, ds, img_len, start_i, end_i)


        elif self.type == '1: slice_sampled_window':

            img_len = self.df.loc[index, 'len']
            center_i = img_len // 2

            ds = self.df.loc[index,'ds']
            if ds == 1:
                ds = ss*ds
            else:
                ds = ss*ds//2

            if ds*con_len <= img_len:
                start_i = center_i - ds*con_len // 2
                end_i = start_i + ds*con_len
                img = data['data'][:, :, start_i:end_i:ds]
                msk = data['mask'][:, :, start_i:end_i:ds]
                img_l = data['data_l'][:, :, start_i:end_i:ds]
                img_s = data['data_s'][:, :, start_i:end_i:ds]
                img_b = data['data_b'][:, :, start_i:end_i:ds]
                #msk1 = data['mask1'][:, :, start_i:end_i:ds]
            else:
                start_i = con_len // 2 - img_len // 2 //ds
                end_i = start_i + img_len//ds
                img[:, :, start_i:end_i] = data['data'][:,:,0:img_len:ds]
                msk[:, :, start_i:end_i] = data['mask'][:,:,0:img_len:ds]
                img_l[:, :, start_i:end_i] = data['data_l'][:, :, 0:img_len:ds]
                img_s[:, :, start_i:end_i] = data['data_s'][:, :, 0:img_len:ds]
                img_b[:, :, start_i:end_i] = data['data_b'][:, :, 0:img_len:ds]
                #msk1 = data['mask1'][:, :, start_i:end_i:ds]
                print(index, ds, img_len, start_i, end_i)

        elif self.type == '2: slice_sampled_aug':

            img_len = self.df.loc[index, 'len']
            center_i = img_len // 2

            ds = self.df.loc[index,'ds']
            ds = ss*ds

            if ds == 1:
                ds = ss*ds
            else:
                ds = ss*ds//2

            if ds*con_len <= img_len:
                start_i = center_i - ds*con_len // 2
                end_i = start_i + ds*con_len
                img = data['data'][:, :, start_i:end_i:ds]
                msk = data['mask'][:, :, start_i:end_i:ds]
                msk1 = data['mask1'][:, :, start_i:end_i:ds]
            else:
                start_i = con_len // 2 - img_len // 2 //ds
                end_i = start_i + img_len//ds
                img[:, :, start_i:end_i] = data['data'][:,:,0:img_len:ds]
                msk[:, :, start_i:end_i] = data['mask'][:,:,0:img_len:ds]
                msk1[:, :, start_i:end_i] = data['mask1'][:, :, 0:img_len:ds]
                print(index, ds, img_len, start_i, end_i)

        elif self.type == '3: slice_avg_window':

            img_len = self.df.loc[index, 'len']
            center_i = img_len // 2

            ds = self.df.loc[index,'ds']
            if ds == 1:
                ds = ss*ds
            else:
                ds = ss*ds//2

            if ds*con_len <= img_len:
                start_i = center_i - ds*con_len // 2
                end_i = start_i + ds*con_len
                img = data['data'][:, :, start_i:end_i:ds]
                msk = data['mask'][:, :, start_i:end_i:ds]
                img_l = data['data_l'][:, :, start_i:end_i:ds]
                img_s = data['data_s'][:, :, start_i:end_i:ds]
                img_b = data['data_b'][:, :, start_i:end_i:ds]
                #msk1 = data['mask1'][:, :, start_i:end_i:ds]
            else:
                start_i = con_len // 2 - img_len // 2 //ds
                end_i = start_i + img_len//ds
                img[:, :, start_i:end_i] = data['data'][:,:,0:img_len:ds]
                msk[:, :, start_i:end_i] = data['mask'][:,:,0:img_len:ds]
                img_l[:, :, start_i:end_i] = data['data_l'][:, :, 0:img_len:ds]
                img_s[:, :, start_i:end_i] = data['data_s'][:, :, 0:img_len:ds]
                img_b[:, :, start_i:end_i] = data['data_b'][:, :, 0:img_len:ds]
                #msk1 = data['mask1'][:, :, start_i:end_i:ds]
                print(index, ds, img_len, start_i, end_i)

        dec = random.choice(range(4))
        if dec == 1 and self.df.loc[index,'valid'] == '0':
            #print('{} is img_auged'.format(index))
            seq = iaa.SomeOf((3, 6), [
                #iaa.Fliplr(0.8),
                #iaa.Flipud(0.8),
                iaa.Multiply((0.8, 1.2)),
                iaa.GaussianBlur(sigma=(0.0, 0.2)),
                iaa.PiecewiseAffine((0.02, 0.06)),
                iaa.Affine(
                    # rotate=(-5, 5),
                    shear=(-5, 5),
                    scale=({'x': (0.8, 1.1), 'y': (0.8, 1.1)})  # to strentch the image along x,y axis
                )
            ])

            seq_det = seq.to_deterministic()

            img = self.img_augmentation(img,seq_det=seq_det)
            img_l = self.img_augmentation(img_l, seq_det=seq_det)
            img_s = self.img_augmentation(img_s, seq_det=seq_det)

        img_c = np.array([img,img_l,img_s,img])
        #print('{} is read'.format(index))
        return img_c

    def lab_generator(self,index):
        cls = {
            '1':'LeftLungAffected',
            '2':'RightLungAffected',
            '3':'CavernsLeft',
            '4':'CavernsRight',
            '5':'PleurisyLeft',
            '6':'PleurisyRight'
        }

        label = []

        for i in range(len(cls)):
            label.append(self.df.loc[index,cls[str(i+1)]])

        return label

    def __getitem__(self, index):

        if self.imgsz == 256:
            img_path = self.df.loc[index, 'data_path']
        elif self.imgsz == 512:
            img_path = self.df.loc[index, 'data_path_512']
        # print(img_path)
        # image = cv2.imread(img_path)
        data = sio.loadmat(img_path)

        image = self.img_generator(index,data,con_len = self.conlen,ss= 1)
        #image = image.reshape(3,48,256,256)
        image = image.transpose(0,3,1,2)

        label = np.array(self.lab_generator(index))
        # label = label.reshape(-1,1)
        # landmarks = landmarks.reshape(-1, 2)
        # sample = {'image': image, 'label': label}

        if self.transform:
            image = self.transform(image)

        if self.debug:
            print('data generator debug:',image.shape)
            plt.imshow(np.squeeze(image[1,24,:,:]))

        return image,label

    @staticmethod
    def parse_csv(csv_file, subset):
        data_frame = pd.read_csv(csv_file)
        data_frame = data_frame[data_frame['valid'] == int(subset)]
        return data_frame


def show_landmarks(image, landmarks):
    """SHow image with landmarks"""
    plt.imshow(image)
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")


if __name__ == "__main__":
    # config = {"aug": True, "subset": 'training', "save_img": True, "add_noise": False}
    # config = {"dataset": 'mammo_calc',"subset": '0'}
    # train_config = dataconfig(**config)
    # train_dataset = DataGenerator(train_config,transform= transforms.ToTensor())
    #
    #
    # train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=4,shuffle= True)
    #
    # num_classes = 1
    # model = resnet2d.ResNet(dataset='calc', depth=34, num_classes=num_classes)
    #
    # criterion = torch.nn.BCELoss().cuda()
    #
    # # print(train_dataloader.batch_size)
    #
    # for i, (images,labels) in enumerate(train_dataloader):
    #     # print(sample['image'])
    #     outputs = model(images)
    #     labels = labels.float().reshape(-1,1)
    #     print(outputs.shape,labels.shape)
    #     loss = criterion(outputs,labels)
    #     print('loss: ',loss)

    valconfig = {"dataset": "tb2020","subset": '0'}
    val_config = dataconfig(**valconfig)
    validation_data = DataGenerator(val_config,transform= None,type='1: slice_sampled')
    #val_loader = DataLoader(validation_data, batch_size=12, num_workers=1, shuffle=True)
    #batch_sampler = batch_sampler(batch_size=6,class_list=range(6))

    val_loader = DataLoader(validation_data,num_workers=1,sampler=BalancedBatchSampler(validation_data,type='multi_label'),batch_size=6)

    for i, (images, labels) in enumerate(val_loader):
        print(i)
        print(labels)
        print(images.shape)

