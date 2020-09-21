# - *- coding: utf- 8 - *-
import numpy as np
import sys
import cv2
import torch
from torch.utils.data import sampler
from torch.utils import data
from torchvision import transforms
import string
import math
import glob
import pickle
import os

class Dataset(data.Dataset):
    def __init__(self, img_file, max_len, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform  
        self.fnames = []
        self.labels = []
        self.max_len = max_len 
        self.num_samples = 0
        self.img_file = img_file
        self.EOS = 'EOS'
        self.PADDING = 'PADDING'
        self.UNK = 'UNK'
        self.voc = self.get_vocabulary()
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        
        img_names = glob.glob(self.img_file+'*.png')
        for name in img_names:
            file_name = name
            label = os.path.basename(file_name)[:-4]
            if len(label) <= self.max_len-1:
                self.fnames.append(file_name)  
                self.labels.append(label)
                self.num_samples += 1

        

    def get_vocabulary(self):
        voc = pickle.load(open('voc.pickle', 'rb'))
        voc.insert(0, 'BK')
        voc.append('EOS')
        voc.append('PADDING')
        voc.append('UNK')
        return voc


    def __getitem__(self, index):
        img = cv2.imread(self.fnames[index])
        rate = int(math.ceil(img.shape[1]/img.shape[0]))    # w/h
        labels = self.labels[index]
        img = self.BGR2GRAY(img)
        # Resize img to 32 x 68
        
        img = cv2.resize(img, (32*rate, 32))
        if img.shape[1] > 100:
            img = cv2.resize(img, (100, 32))
        else:
            img = cv2.copyMakeBorder(img, 0, 0, 0, 100-img.shape[1], cv2.BORDER_CONSTANT, value=255)
        
        img = self.transform(img)
        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
        label_list = []
        
        for char in labels:
            if char in self.char2id:
                label_list.append(self.char2id[char])
            else:
                #print('{0} is out of vocabulary'.format(char))
                label_list.append(self.char2id[self.UNK])    
        label_list = label_list + [self.char2id[self.EOS]] 
        length = len(label_list)
        label[:length] = np.array(label_list)
        label = torch.LongTensor(label)
        return img, label, length

    def BGR2GRAY(self, img):
      return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def __len__(self):
      return self.num_samples 

if __name__ == '__main__':
    train = Dataset(sys.argv[1], 17, transforms.ToTensor())
    train.__getitem__(0)
