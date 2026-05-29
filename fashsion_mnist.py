import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch as ts 
import torch.nn as nn
import pandas as pd 
import numpy as np 

# transforming the images to tensor and normalizing it 
from torchvision import transforms

trans = transforms.Compose([
    transforms.ToTensor(), # image to tensor to scale(0,1)
    transforms.Normalize((0.5,),(0.5,)) # set of values for normalizing( -1 , 1)
])

# importing dataset 
from torchvision.datasets import FashionMNIST

data_train = FashionMNIST( root = "./data_Fashsion", train=True ,download=True , transform=trans)
data_test = FashionMNIST(root = "./data_Fashsion" , train=False , download=True , transform=trans )

# data splittinng for train and test and LOADING]
from torch.utils.data import DataLoader , random_split

train_size = int(0.8*len(data_train))
val_size = len(data_train) - train_size

train_dataset , val_dataset = random_split( data_train , [train_size , val_size])

# loading

x_train = DataLoader(train_dataset , batch_size=64 , shuffle=True)
x_val = DataLoader(val_dataset , batch_size=64  , shuffle=False)
x_test = DataLoader(data_test , batch_size=64 , shuffle=False)

# model 

class fashsion(nn.Module()):

    def __init__(self):
        super(fashsion , self).__init__()

        # flattening
        self.flatten = nn.Flatten()
 
        # hidden layer
        self.hiddenlayer1 = nn.Linear(784 , 16)
        
        #cleft branches
        self.Leftlayer1=nn.Linear(16 ,8)
        self.Leftlayer2=nn.Linear(8 ,8)
        
        # right branches
        self.rightlayer1=nn.Linear(16,12)
        self.rightlayer2=nn.Linear (12, 8)

        #activation
        self.relu = nn.ReLU()

        #output
        self.output = nn.Linear(16 , 10)   

    def forward(self , x):

        # flattening
        x = self.flatten(x) # (28*28 , 784)

        # hiddenlayer1
        h1 = self.relu(self.hiddenlayer1(x)) # (784 , 16)

        # LEFT LAYERS 
        Lh1 = self.relu(self.hiddenlayer1(h1)) # (16 ,8)
        Lh2 = self.relu(self.hiddenlayer1(Lh1)) # (8 ,8)
        skip= Lh1 + Lh2 # (8 ,8)

        # Right layers
        Rh1 = self.relu(self.hiddenlayer1(h1)) # (16 , 12) 
        Rh2 = self.relu(self.hiddenlayer1(Rh1)) #(12 ,8)

        # concantenate
        con = ts.concatenate([skip , Rh2] , dim=1) # (8*8 , 16) ; [batch , 16]

        out = self.output(con)        

        return out
    



        



        
        




