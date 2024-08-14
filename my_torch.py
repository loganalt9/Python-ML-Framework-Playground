import numpy as np
import torch

"""Tensors
data = [[1,2],[3,4]]

x_data = torch.tensor(data)

print(x_data)

shape = (3,2,)

rand_tensors = torch.rand(shape)

print(rand_tensors)

print (x_data.device)
"""
"""Datasets & Dataloaders"""
from torch.utils.data import DataLoader, Dataset


class CustomDataSet(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data.float()
        self.y_data = y_data.float()
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        features = self.x_data[idx]
        labels = self.y_data[idx]

        return features,labels
    

x = torch.randint(-10, 10,(20000,2)) / 10.0
y = torch.sum(x,dim=1, keepdim=True)

dataset = CustomDataSet(x,y)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

train_iter = iter(dataloader)

"""
Transforms good for image data or converting into a data form that pytorch can handle
Pytorch can already handle lists/numpy ndarrays
"""
from torchvision.transforms import Lambda, ToTensor

"""Nueral network"""
from torch import nn

#Find device
device = "cpu"
# This is labeled a module in flight
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2,5),
            nn.ReLU(),
            nn.Linear(5,1)
        )
    
    def forward(self, x):
        return self.model(x)
    
model = NeuralNetwork().to(device)

"""Optimizing model parameters/train loop"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

size = len(dataloader)
model.train()

records = []
for batch, (x,y) in enumerate(dataloader):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 100 == 0:
        loss, current = loss.item(), batch * dataloader.batch_size + len(x)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        records.append({"train/loss": loss, "epoch": batch})

test_input = torch.tensor([5.0,6.0]) / 10.0
predicted_output = model(test_input).item()

#print(predicted_output*10)

df = pd.DataFrame.from_records(records)
sns.lineplot(df, x="epoch", y="train/loss")
plt.show()
