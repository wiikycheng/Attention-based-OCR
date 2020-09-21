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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
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
      
        loss, rec_pred = model(img, label, length)
     
        for idx, l in enumerate(label):
            ans = l[:length[idx]]
            pred = torch.argmax(rec_pred[idx], dim=1)[:length[idx]]
            t_ans = ''
            p_ans = ''
            for index in range(length[idx]):
                t_ans += id2char[ans[index].item()]
                p_ans += id2char[pred[index].item()]
            print(t_ans)
            print(p_ans)
            to_pil = torchvision.transforms.ToPILImage()
            rec_img = img.detach()
            rec_img = rec_img[idx, :, :,:].cpu().numpy()*255
            rec_img = rec_img.transpose(1, 2, 0)
            rec_img = to_pil(rec_img)
            plt.imshow(rec_img, cmap='gray', vmin=0, vmax=255)
            plt.show()
              
        
            
