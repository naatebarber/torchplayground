import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class LR(torch.nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LR()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    pred_y = model(x_data)
    loss = criterion(pred_y, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("epoch {}, loss {}".format(epoch, loss.item()))

new_var = Variable(torch.Tensor([[4.0]]))
# Should be 8
pred_y = model(new_var).item()
print(f"{pred_y} should be close to 8")
