import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self,indim=28*28,hidden=512,outdim=10,activate = 'relu'):
        super(mlp, self).__init__()
        self.fc1   = nn.Linear(indim, hidden)
        self.fc2   = nn.Linear(hidden, outdim)
        self.activate = activate
        if self.activate == 'relu':
            self.act = F.relu
        elif self.activate == 'softmax':
            self.act = F.softmax()



    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        return out
