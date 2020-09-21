import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dataset_jp import Dataset
from build_s2s import ModelBuilder 
import sys
import time


def train(num_poch):
    use_gpu = torch.cuda.is_available()
    learning_rate = 0.01
    batch_size = 16
    epoch_nums = num_poch

    train_dataset = Dataset(img_file=sys.argv[1], max_len=17, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = Dataset(img_file=sys.argv[2], max_len=17, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    num_classes = len(train_dataset.voc)

    model = ModelBuilder(rec_num_classes=num_classes, sDim=512, attDim=512, max_len_labels=17)
    
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    
    if use_gpu:
        model.cuda()
       
    
    for epoch in range(epoch_nums):
        end = time.time()
        total_train_loss = 0
        model.train()
        
        for i, (img, label,  length) in enumerate(train_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
                length = length.cuda()
            
            loss, _ = model(img, label, length)
            #print(loss)
            total_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print('\rEpoch [%d/%d], Training loss: %.4f' % (epoch+1, epoch_nums, total_train_loss/len(train_loader)), end='\n')

        
        
        total_test_loss = 0 
        acc_sum = 0
        model.eval()
        for i, (img, label, length) in enumerate(test_loader):
            label = torch.LongTensor(label)
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
                length = length.cuda()
            
            loss, rec_pred = model(img, label, length)
            total_test_loss += loss.item()
            
        print('\rEpoch [%d/%d], Testing loss: %.4f' % (epoch+1, epoch_nums, total_test_loss/len(test_loader)), end='\n')
        print('\n')
        
        save_path = sys.argv[3]
        torch.save(model.state_dict(), save_path)


