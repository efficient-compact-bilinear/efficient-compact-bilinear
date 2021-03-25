import torch
from torch.nn import functional as F
from torch import nn
import torch.cuda
from torch.autograd import Variable
import gc
class runLayer(nn.Module):
    def __init__(self):
        super(runLayer, self).__init__()
        self.iter = 2
    def forward(self,input):
        sz = input.size()
        input = input.view(sz[0],512,-1)
        v = Variable(torch.rand(sz[0],input.size(2),1).type(torch.cuda.FloatTensor),requires_grad=False)
        eps = 1e-5
        A = input
        B = A.permute(0,2,1)
        for i in range(self.iter):
           v = A.bmm(v) 
           v = B.bmm(v)
        vNorm = v.mul(v).sum(1,keepdim=True).add(eps).sqrt().expand(v.size())
        v = v.div(vNorm)
        u = input.bmm(v)
        bLgMat = u.bmm(v.permute(0,2,1))
        A = A.sub(bLgMat.mul(0.6))
        output = A
        return output;


