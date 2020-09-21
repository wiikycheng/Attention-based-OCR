import torch
import torchvision.transforms as transforms
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataset_jp import Dataset
from build_s2s import ModelBuilder
import sys
import time
import matplotlib.pyplot as plt

def test():
    use_gpu = torch.cuda.is_available()
    batch_size = 4
  
    test_dataset = Dataset(img_file=sys.argv[1], max_len=17, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    num_classes = len(test_dataset.voc)
    model = ModelBuilder(rec_num_classes=num_classes, sDim=512, attDim=512, max_len_labels=17)
    id2char = dict(zip(range(len(test_dataset.voc)), test_dataset.voc))

    # attDim = 256
    path = sys.argv[2]
    state = torch.load(path)


    model.load_state_dict(state)
    if use_gpu:
        model.cuda()
    
    acc_sum = 0
    model.eval()
    for i, (img, label, length) in enumerate(test_loader):
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
            length = length.cuda()
      
        input_lengths = torch.zeros((batch_size), dtype=torch.int64).fill_(35)

        loss, rec_pred = model(img, label, length)
     
        for idx, l in enumerate(label):
            ans = l[:length[idx]]
            pred = torch.argmax(rec_pred[idx], dim=1)[:length[idx]]
            if torch.equal(ans, pred):
                acc_sum += 1
                
            
      
    final_acc = acc_sum/batch_size
    print('Accuracy: ', final_acc/len(test_loader))
      
      
