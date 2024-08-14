import torch

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import CIFAR10

import matplotlib.pyplot as plt
import seaborn as sbs
import pandas as pd

records = []

train_data = CIFAR10('./cifar',train=True,transform=transforms.ToTensor())
val_data = CIFAR10('./cifar',train=True,transform=transforms.ToTensor())
test_data = CIFAR10('./cifar',train=False,download=False,transform=transforms.ToTensor())


train_loader = DataLoader(train_data)
test_loader = DataLoader(test_data)
val_loader = DataLoader(val_data)

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        """
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(32*32*3,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        """
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    
    def forward(self, x):
        """
        x = self.flatten(x)
        x = self.model(x)
        return x
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

module = MyModule()

optimizer = optim.SGD(module.parameters(), lr=0.001)

#Training
for epoch in range(3):
    module.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = module(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 0 and i != 0:
            records.append({'batch': (i)+(epoch*60_000), 'type': 'train/loss', 'value': running_loss/1000})
            running_loss = 0.0
            
    
    module.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            outputs = module(inputs)

            predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted[1] == labels).sum().item()

            if i % 1000 == 0 and i != 0:
                records.append({'batch': (i)+(epoch*60_000), 'type': 'val/acc', 'value': correct/total})
                correct = 0
                total = 0


#Testing
correct = 0
total = 0
module.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader):
        image, label = data

        output = module(image)

        _, prediction = torch.max(output.data,1)

        total += label.size(0)
        correct += (prediction == label).sum().item()

print(f'Accuracy on the test images: {100 * correct // total} %')

df = pd.DataFrame.from_records(records)
ax = sbs.lineplot(data=df, x='batch', y='value', hue='type')
ax.set_title('Torch')
plt.show()