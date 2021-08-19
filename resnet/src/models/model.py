import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

class BaseBlock(nn.Module):
  expansion=1
  def __init__(self,in_planes,planes,stride=1,dimens=None):
    super(BaseBlock,self).__init__()
    """
     convolutional layers with Batch norms
    """
    self.conv1 = nn.Conv2d(in_planes,planes,stride = stride,kernel_size=3,padding=1)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes,planes,stride=1,kernel_size=3,padding=1)
    self.bn2 = nn.BatchNorm2d(planes)
    self.dimens = dimens

  def forward(self,x):
    res = x
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))

    if self.dimens is not None:
      res = self.dimens(res)

    out +=res 
    out = F.relu(out)
    return out



class BottleNeck(nn.Module):
  expansion=4
  def __init__(self,in_planes,planes,stride=1,dimens=None):
  
    super(BottleNeck,self).__init__()
    self.conv1 = nn.Conv2d(in_planes,planes,stride=1,kernel_size=1)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes,planes*self.expansion,kernel_size=1)
    self.bn3 = nn.BatchNorm2d(planes*self.expansion)
    self.dimens = dimens



  def forward(self,x):
    res = x
    out =F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))

    if self.dimens is not None:
      res = self.dimens(res)
    out +=res
    out =F.relu(out)
    return out



class ResNet(nn.Module):
  def __init__(self,block,num_layers,classes=10):
    super(ResNet,self).__init__()
    self.in_planes = 64
    self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
    self.bn1 = nn.BatchNorm2d(64)

    self.layer1 = self._make_layer(block,64,num_layers[0],stride=1)
    self.layer2 = self._make_layer(block,128,num_layers[1],stride=2)
    self.layer3 = self._make_layer(block,256,num_layers[2],stride=2)
    self.layer4 = self._make_layer(block,512,num_layers[3],stride=2)
    self.averagepool = nn.AvgPool2d(kernel_size=4,stride=1)
    self.fc = nn.Linear(512*block.expansion,classes)


  def _make_layer(self,block,planes,num_layers,stride=1):
    dimens = None
    if stride !=1 or planes != self.in_planes*block.expansion:
      dimens = nn.Sequential(
          nn.Conv2d(self.in_planes,planes*block.expansion,kernel_size=1,stride=stride),
          nn.BatchNorm2d(planes*block.expansion)
      )

    netlayers = []
    netlayers.append(block(self.in_planes,planes,stride=stride,dimens=dimens))
    self.in_planes = planes*block.expansion
    for i in range(1,num_layers):
      netlayers.append(block(self.in_planes,planes))
      self.in_planes = planes*block.expansion
    return nn.Sequential(*netlayers)


  def forward(self,x):
    x = F.relu(self.bn1(self.conv1(x)))

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = F.avg_pool2d(x,4)
    x = x.view(x.size(0),-1)#convert from 3d to 2d
    x = self.fc(x)
    return x

