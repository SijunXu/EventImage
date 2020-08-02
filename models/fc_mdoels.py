import torch
import torch.nn as nn
import torch.nn.functional as F

class J2_net(nn.Module):
    def __init__(self, layers, indim=8, nb_fw=50, BN=False):
        super(J2_net, self).__init__()
        def com_layer(d_in, d_out, BN=False):
            if BN:
                return nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(inplace=True), nn.BatchNorm1d(d_out))
            else:
                return nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(inplace=True))
            
        self.layers = nn.ModuleList([
            com_layer(indim, layers[i], BN) if i==0
            else com_layer(layers[i-1], layers[i], BN) for i in range(len(layers))
        ])
        self.fwlayers = nn.ModuleList([
            com_layer(nb_fw, layers[i], BN) if i==0
            else com_layer(layers[i-1], layers[i], BN) for i in range(len(layers))
        ])
        self.classify = nn.Sequential(nn.Linear(layers[-1], 1), nn.Sigmoid())

    def forward(self, x, fw):
        for l in self.layers:
            x = l(x)
        for l in self.fwlayers:
            fw = l(fw)
        out = self.classify(x+fw)
        return out
