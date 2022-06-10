import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data as utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import IPython
import time
import IPython.display as display


# Dataset specification
n = 1000
dim = 1
std = 10.0
# Train specification
epochs = 100
lr = 0.0001
bs = 100
log_interval = 10

from torchvision.utils import make_grid as make_grid
#from PIL import Image
import requests
from io import BytesIO
import cv2
from resizeimage import resizeimage

bunny = np.load('bunny.npy')
bunny = bunny
#bunny = Image.open('/content/bunny04.png')#BytesIO(response.content))
#bunny = np.uint8(np.array(bunny))
#bunny = bunny//255
#bunny = np.abs(bunny-1)
#bunny = np.pad(bunny,((54,54),(34,34)),mode='constant')

cow = np.load('cow.npy')
cow = cow
#cow = Image.open('/content/cow1.png') #BytesIO(response.content))
#cow = np.uint8(np.array(cow))
#cow = cv2.resize(cow, (90,100), interpolation = cv2.INTER_AREA)
#cow = cow//255
#cow = np.abs(cow-1)
#cow = np.pad(cow,((47,47),(40,40)),mode='constant')

num_rows, num_cols = bunny.shape[:2]

# bottom right: 46 39 to 60 70
translation_matrix = np.float32([ [1,0,0], [0,1,0] ])
img_translation = cv2.warpAffine(cow, translation_matrix, (num_cols, num_rows))
plt.imshow(img_translation,cmap='gray')



biased_train_bun = []
biased_train_cow = []

for i in range(1000):
  
  x_val = np.random.randint(46,60)
  y_val = np.random.randint(39,70)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
  biased_train_bun.append(im)
  x_val = np.random.randint(-53,-39)
  y_val = np.random.randint(-65,-34)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
  biased_train_bun.append(im)
  
  y_val = np.random.randint(44,64)
  x_val = np.random.randint(-46,-22)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(cow, translation_matrix, (200, 200))
  biased_train_cow.append(im)
  y_val = np.random.randint(-56,-33)
  x_val = np.random.randint(55,78)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(cow, translation_matrix, (200, 200))
  biased_train_cow.append(im)
  
unbiased_train_bun = []
unbiased_train_cow = []

for i in range(2000):
  
  x_val = np.random.randint(-53,60)
  y_val = np.random.randint(-65,70)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(bunny, translation_matrix, (200, 200))
  unbiased_train_bun.append(im)
  
  y_val = np.random.randint(-56,64)
  x_val = np.random.randint(-46,78)
  translation_matrix = np.float32([ [1,0,x_val], [0,1,y_val] ])
  im = cv2.warpAffine(cow, translation_matrix, (200, 200))
  unbiased_train_cow.append(im)
  
biased_train_cow = np.array(biased_train_cow)
biased_train_bun = np.array(biased_train_bun)
unbiased_train_cow = np.array(unbiased_train_cow)
unbiased_train_bun = np.array(unbiased_train_bun)





"""# New Section"""

unbiased_squares = unbiased_train_cow
unbiased_hearts = unbiased_train_bun
biased_squares = biased_train_cow
biased_hearts = biased_train_bun

some_unbiased_squares = biased_squares[:25]
np.random.shuffle(some_unbiased_squares)



grid = make_grid(torch.from_numpy(np.float32(some_unbiased_squares)).view(25,1,200,200),nrow=5,pad_value=1,padding=10)
plt.axis('off')
plt.imsave('biased_stars.png',grid.numpy().transpose(1,2,0))
plt.imshow(grid.numpy().transpose(1,2,0))

unbiased_data = np.concatenate((unbiased_squares,unbiased_hearts))
unbiased_data = torch.from_numpy(np.float32(unbiased_data))
labels = np.concatenate((np.zeros(len(unbiased_squares)),np.ones(len(unbiased_squares))))
labels = torch.from_numpy(labels).long()
biased_data = np.concatenate((biased_squares,biased_hearts))
biased_data = torch.from_numpy(np.float32(biased_data))

train_data = utils.TensorDataset(biased_data,labels) 
train_loader = utils.DataLoader(train_data,batch_size=100,shuffle=True) 
test_data = utils.TensorDataset(unbiased_data,labels) 
test_loader = utils.DataLoader(test_data,batch_size=100)

# Build FC net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(200*200, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.c1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.c2 = nn.Conv2d(32, 32, 5, 1, padding=2)
        self.c3 = nn.Conv2d(32, 32, 5, 1, padding=2)
        self.pool = nn.AvgPool2d(200)
        self.fc = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
      
      
class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.c1 = nn.Conv2d(1, 128, 5,stride=2,padding=2)#, 2, padding=2)
        self.c2 = nn.Conv2d(128, 128, 5,stride=2,padding=2)#, 2, padding=2)
        self.c3 = nn.Conv2d(128, 128, 5,stride=2,padding=2)#, 2, padding=2)
        self.pool = nn.AvgPool2d(25)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#criterion = nn.MSELoss()     
criterion = nn.CrossEntropyLoss()

lr = 0.001

epochs = 5

for epoch in range(epochs):
    total = 0.
    correct = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        #data, target = Variable(data).cuda(), Variable(target).cuda()
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 200*200)
        target = target.squeeze()
        optimizer.zero_grad()
        net_out = net(data).squeeze()
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                           bs * batch_idx / len(train_loader), loss.item()))
            
        _, predicted = torch.max(net_out.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Train Acc: '+str(correct/total))

total = 0.
correct = 0.
for idx, (data, target) in enumerate(test_loader):
  #data, target = Variable(data).cuda(), Variable(target).cuda()
  data, target = Variable(data), Variable(target)
  data = data.view(-1, 200*200)
  optimizer.zero_grad()
  net_out = net(data).squeeze()
  _, predicted = torch.max(net_out.data, 1)
  total += target.size(0)
  correct += (predicted == target).sum().item()
print('Test Acc: '+str(correct/total))


