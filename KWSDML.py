import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


class DML(nn.Module):

    def __init__(self):
        super(DML, self).__init__()
        self.project = nn.Sequential(
           
            nn.Linear(36,36),
            nn.ReLU(inplace=True),   
            nn.Linear(36,20)
        )

        self.bias = torch.nn.Parameter(torch.Tensor([0]))
        self.out = nn.Sigmoid()


    def forward_one(self, x):
        x = self.project(x)
        x = x.view(x.size(0), -1)    
        return x

    def forward(self, x1, x2):
        out1 = torch.squeeze(self.forward_one(x1))
        out2 = torch.squeeze(self.forward_one(x2))
        out = torch.diag(torch.matmul(out1, out2.T))
        out = self.out(out + self.bias)
        return out


def get_batch(data, targets, batch_size=32):
    
    K = len(set(np.squeeze(targets)))

    
    anchor = np.zeros((batch_size, data.shape[1]))
    friend = np.zeros((batch_size, data.shape[1]))
    foe = np.zeros((batch_size, data.shape[1]))
    
    for k in range(batch_size):
        
        c1, c2 = random.sample(range(K), 2)
        friend_clas_indices = np.where(targets == c1)[0]
        foe_clas_indices = np.where(targets == c2)[0]
    
        anchor_inx, friend_inx = random.sample(list(friend_clas_indices), 2)
        foe_inx = random.sample(list(foe_clas_indices),1) [0]
        anchor[k, :] = data[anchor_inx,:]
        friend[k, :] = data[friend_inx,:]
        foe[k, :] = data[foe_inx,:]
        
    friend_label = np.ones(batch_size)
    foe_label = np.zeros(batch_size)
    
    batch_1 = torch.from_numpy(np.concatenate((anchor, anchor), axis = 0)).cuda()
    batch_2 = torch.from_numpy(np.concatenate((friend, foe),  axis = 0)).cuda()
    label = torch.from_numpy(np.concatenate((friend_label, foe_label))).cuda()

    return batch_1, batch_2, label

class MaskedXentMarginLoss:
    def __init__(self, zero_weight=1.0, one_weight=1., margin=0.7, eps=1e-7, **kwargs):
        self.margin = margin
        self.zero_weight = zero_weight
        self.one_weight = one_weight
        self.eps = eps
        
    def __call__(self, pred, labels):
        pred = torch.clamp(pred, min=self.eps, max=1-self.eps)
        loss_mat = -self.one_weight * labels * torch.log(pred) * (pred <= self.margin).float()
        loss_mat -= self.zero_weight * (1 - labels) * torch.log(1-pred) * (pred >= 1 - self.margin).float()
        mask = ((labels == 0) + (labels == 1))
        loss = loss_mat[mask].sum()
        return loss
    

def calculate_scores(data, medoids, targets):
    classes = set(np.squeeze(targets))                        
    K = len(classes)           
    correct = 0
    incorrect = 0
    for i in range(len(data)):
        x = data[i,:]
        _max = -10000
        _selection = 0
        for medoid in range(K):
            y = medoids[:, medoid]
            sim = np.dot(x, y)
            if sim > _max:
                _selection = medoid
                _max = sim
        if _selection == targets[i]:
            correct +=1
        else:
            incorrect +=1
    return correct/(correct+incorrect)
        