import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt


def NormalizedConvLayer(in_channels, out_channels, kernel_size, padding = 0, stride = 1, batch_norm = True):
    """
    Returns Convolutional Layer with batch-norm 
    """
    bias = not batch_norm
    layers = []
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = bias)
    layers.append(conv)
    if batch_norm:
        norm = nn.BatchNorm2d(out_channels)
        layers.append(norm)
    
    return nn.Sequential(*layers)   


class DuellingDQN(nn.Module):
    def __init__(self, in_channels, conv_dim = 32, kernel_size = 6, num_outputs = 5):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.conv_dim = conv_dim
        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        
        # input is an image a 464 x 464 image
        
        self.conv1 = NormalizedConvLayer(in_channels, conv_dim, kernel_size, stride = 4, padding = 1)
        # default output shape = 32 x 116 x 116   
        
        self.conv2 = NormalizedConvLayer(conv_dim, 2*conv_dim, kernel_size, stride = 4, padding = 1)
        # default output shape = 64 x 29 x 29 
        
        self.conv3 = NormalizedConvLayer(2*conv_dim, 4*conv_dim, kernel_size, stride = 4, padding = 1)
        # default output shape = 128  x 7 x 7 
        
        self.conv4 = NormalizedConvLayer(4*conv_dim, 8*conv_dim, kernel_size = 4, stride = 2, padding = 1)
        # default output shape = 256 x 3 x 3 
        
        self.conv5 = NormalizedConvLayer(8*conv_dim, 16*conv_dim, kernel_size = 3)
        # default output shape = 512 x 1 x 1 
        
        self.value_in = nn.Linear(in_features = 16*conv_dim, out_features = 8*conv_dim, bias = False)
        self.BatchNormV = nn.BatchNorm1d(num_features = 8*conv_dim)
        self.value_out = nn.Linear(in_features = 8*conv_dim, out_features = 1)
        
        self.advantage_in = nn.Linear(in_features = 16*conv_dim, out_features = 8*conv_dim, bias = False)
        self.BatchNormA = nn.BatchNorm1d(num_features = 8*conv_dim)
        self.advantage_out = nn.Linear(in_features = 8*conv_dim, out_features = num_outputs)
        
        
    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.shape)
        x = F.selu(self.conv1(x))
        # print(x.shape)
        x = F.selu(self.conv2(x))
        # print(x.shape)
        x = F.selu(self.conv3(x))
        # print(x.shape)
        x = F.selu(self.conv4(x))
        # print(x.shape)
        x = F.selu(self.conv5(x))
        # print(x.shape)
        
        assert x.shape == (batch_size, 16*self.conv_dim, 1, 1), \
        "Wrong shape of conv5 output. Expected {}, got {}".format((batch_size, 16*self.conv_dim, 1, 1), x.size())
        
        x = x.view(batch_size, -1)
        # print(x.shape)
        
        v = F.selu(self.BatchNormV(self.value_in(x)))
        v = self.value_out(v)
        # print(v.shape)
        
        a = F.selu(self.BatchNormA(self.advantage_in(x)))
        a = self.advantage_out(a)
        # print(a.shape)
        
        mn = a.mean(dim = 1, keepdim = True)
        # print(mn.shape)
        
        return v + (a - mn)   
    
    
    def __repr__(self):
        return "DuellingDQN(in_channels = {}, conv_dim = {},"\
    " kernel_size = {}, num_outputs = {})".format(self.in_channels, self.conv_dim, self.kernel_size, self.num_outputs)