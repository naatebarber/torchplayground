import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from sklearn.datasets import load_iris
from random import randint
import pandas as pd

ds = load_iris(as_frame=True, return_X_y=True)
x_data = ds[0].to_numpy(dtype=np.double)
y_data = ds[1].to_numpy(dtype=np.double)

def get_xy(x_data, y_data):
  ix = randint(0, len(x_data) - 1)
  return (
    Variable(torch.Tensor(x_data[ix])), 
    Variable(torch.Tensor([y_data[ix]]).float())
  )

class MLR(torch.nn.Module):
  def __init__(self):
    super(MLR, self).__init__()
    self.linear = torch.nn.Linear(4, 1)
    self.type = torch.float64

  def forward(self, x):
    y_pred = self.linear(x)
    return y_pred

model = MLR()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# train

for e in range(10000):
  ipt, label = get_xy(x_data, y_data)
  pred_y = model(ipt)
  loss = criterion(pred_y, label)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  print("epoch {}, loss {}".format(e, loss.item()))

# test
stat = []
for i in range(500):
  ipt, label = get_xy(x_data, y_data)
  print(ipt, label)
  pred = model(ipt).item()
  rounded = round(pred)
  stat.append(1 if rounded == label else 0)

acc = (sum(stat) / len(stat)) * 100 
print(f"{acc}% rounded accuracy.")
