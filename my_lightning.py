from typing import Any
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.optim import SGD
from torch.nn.functional import cross_entropy
from torch import nn
from torch.utils.data import DataLoader

class MyModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.hidden_activation = nn.ReLU()

        self.records = []
        self.train_count = 0
        self.val_count = 0
        self.test_count = 0

        self.running_loss = 0.0
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.hidden_activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.hidden_activation(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.hidden_activation(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.running_loss = 0.0
            
        self.train_count += 1
        inputs, labels = batch
        prediction = self(inputs)
        loss = cross_entropy(prediction, labels)

        self.running_loss += loss.item()
        if batch_idx % 100 == 0 and batch_idx != 0:
            self.records.append({'value': self.running_loss / 100.0, 'batch': batch_idx + (self.current_epoch * 782), 'type': 'train/loss'})
            self.running_loss = 0.0
        return loss

    def test_step(self, batch, batch_idx):
        self.test_count += 1
        inputs, labels = batch
        prediction = self(inputs)
        loss = cross_entropy(prediction, labels)

        _, prediction = torch.max(prediction, 1)
        test_acc = (labels == prediction).float().mean().item()

        #self.records.append({'value': test_acc, 'batch': batch_idx, 'type': 'test/acc'})
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_count += 1
        inputs, labels = batch
        predicition = self(inputs)
        loss = cross_entropy(predicition, labels)

        _, predicition = torch.max(predicition, 1)
        val_acc = (labels == predicition).float().mean().item()

        if self.val_count % 100 == 0:
            self.records.append({'value': val_acc, 'batch': batch_idx + (self.current_epoch*782), 'type': 'val/acc'})
        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.07)
        return optimizer
    


module = MyModule()


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = CIFAR10('./cifar', train=True, transform=transform)
testset = CIFAR10('./cifar', train=False, transform=transform)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
val_loader = DataLoader(trainset, batch_size=64, shuffle=False)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

trainer = L.Trainer(max_epochs=5)
trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.test(model=module, dataloaders=test_loader)

df = pd.DataFrame.from_records(data=module.records)
ax = sbs.lineplot(data=df, x='batch', y='value', hue='type')
ax.set_title('Lightning')
plt.show()