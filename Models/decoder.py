from __future__ import absolute_import
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init

class AttentionRecognitionHead(nn.Module):
    '''
        input : [b x 16 x 64 x in_planes] [encoder_feature, rec_targets, rec_lengths]
        output : probability sequence : [b x T x num_classes]  --> [T x b x num_classes]
    '''
    def __init__(self, num_classes, in_planes, sDim, attDim, max_len_labels):
        super(AttentionRecognitionHead, self).__init__()
        self.num_classes = num_classes    # include the <EOS>
        self.in_planes = in_planes
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels
        self.embedd_size = 64
        self.charEmbed = nn.Embedding(num_classes, self.embedd_size)
        self.decoder = DecoderUnit(sDim=sDim, xDim=in_planes, yDim=num_classes, attDim=attDim, len_voc=num_classes, embeddDim =self.embedd_size)
        self.use_cuda = torch.cuda.is_available()

    def forward(self, x, prev_hidden):
        batch_size = x.size(0)
        
        state = prev_hidden.unsqueeze(0)
            
        outputs = []

        for i in range(self.max_len_labels):
            if i == 0:
                if self.use_cuda :
                    y_prev = self.charEmbed(torch.zeros(batch_size,dtype=torch.long).cuda())
                else :
                    y_prev = self.charEmbed(torch.zeros(batch_size,dtype= torch.long))
            else:
                y_prev = outputs[i-1]
                max_id = torch.argmax(y_prev, dim=1)
                y_prev = self.charEmbed(max_id)

            output, state = self.decoder(x, state, y_prev)
            outputs.append(output)
        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)   # [b x T x num_classes]
        outputs = outputs.transpose(1, 0)   # [T x b x num_classes]
        log_probs = F.log_softmax(outputs, dim=2)

        return log_probs



class AttentionUnit(nn.Module):
    def __init__(self, sDim, xDim, attDim):
        super(AttentionUnit, self).__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim
        self.sEmbed = nn.Linear(sDim, attDim)
        self.xEmbed = nn.Linear(xDim, attDim)
        self.wEmbed = nn.Linear(attDim, 1)

    def forward(self, x, sPrev):
        '''
            x : input feature sequence [batch_size x T x in_planes]
            sPrev : previous internal state   [1 x batch_size x sDim]
        '''
        batch_size, T, _ = x.size()   # (b, T, xDim), xDim : in_planes
        x = x.view(-1, self.xDim)
        
        xProj = self.xEmbed(x)    # [(b x T) x attDim]
        xProj = xProj.view(batch_size, T, -1)  # [b x T x attDim]
        sPrev = sPrev.squeeze(0)
        sProj = self.sEmbed(sPrev)    # [b x attDim]
        sProj = torch.unsqueeze(sProj, 1)   # [b x 1 x attDim]
        sProj = sProj.expand(batch_size, T, self.attDim)
       
        tanh_sum = torch.tanh(sProj + xProj)
        tanh_sum = tanh_sum.view(-1, self.attDim)

        eProj = self.wEmbed(tanh_sum)   # [(b x T) x 1]
        eProj = eProj.view(batch_size, T)


        alpha = F.softmax(eProj, 1)  # attention weights for each batch

        return alpha

class DecoderUnit(nn.Module):
    def __init__(self, sDim, xDim, yDim, attDim, len_voc, embeddDim):
        super(DecoderUnit, self).__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.yDim = yDim
        self.attDim = attDim
        self.embDim = attDim
       
        self.attention_unit = AttentionUnit(sDim, xDim, attDim)
    
        self.gru = nn.GRU(input_size=xDim + embeddDim, hidden_size=sDim, batch_first=True)
        self.fc = nn.Linear(sDim, yDim)


    def forward(self, x, sPrev, yPrev):
        '''
        x : feature sequence from the image decoder [batch_size x T x in_planes]
        sPrev : previous internal state   [1 x batch_size x sDim]
        yPrev : previous output   [batch_size]
        '''
        batch_size, T, _ = x.size()
        alpha = self.attention_unit(x, sPrev)

        '''
         alpha : [batch_size x T]  --> [batch_size x 1 x T]
         x : [batch_size, T, in_planes]
         after bmm : [batch_size x 1 x T] --> [batch_size x T]

        '''
        context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)  # [batch_size x T]

        output, state = self.gru(torch.cat([yPrev, context], 1).unsqueeze(1), sPrev)
        output = output.squeeze(1)
        output = self.fc(output)

        return output, state



if __name__ == '__main__': 
    x = torch.rand((3, 35, 512))
    prev = torch.rand((3, 512))
    model =  AttentionRecognitionHead(79, 512, 512, 512, 17)
    output = model(x, prev)
    #print(output.size())
      
