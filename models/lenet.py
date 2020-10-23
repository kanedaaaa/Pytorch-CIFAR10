import torch
import torch.nn as nn

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()

    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)

    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2,2)

    self.f1 = nn.Linear(16*5*5, 120)
    self.f2 = nn.Linear(120, 84)
    self.f3 = nn.Linear(84, 10)
  
  def forward(self, x):
    out = self.pool(self.relu(self.conv1(x)))
    out = self.pool(self.relu(self.conv2(out)))

    out = out.view(-1, 16*5*5)

    out = self.relu(self.f1(out))
    out = self.relu(self.f2(out))
    pred = self.relu(self.f3(out))

    return pred  



