# MNIST 
Handwritten digit classifier using PyTorch.



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=32, shuffle=False)
```


```python
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])),
    batch_size=32, shuffle=False)

```


```python
class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.net = nn.Linear(28 * 28, 10)
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.net(x)
        return F.softmax(output)
```


```python
model = BasicNN()
optimizer = optim.SGD(model.parameters(), lr=0.001)
```


```python
def test():
    total_loss = 0
    correct = 0
    for image, label in test_loader:
        image, label = Variable(image), Variable(label)
        output = model(image)
        total_loss += F.cross_entropy(output, label)
        correct += (torch.max(output, 1)[1].view(label.size()).data == label.data).sum()
    total_loss = total_loss.data[0] / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return total_loss, accuracy

```


```python
def train():
    model.train()
    for image, label in train_loader:
        image, label = Variable(image), Variable(label)
        optimizer.zero_grad()
        output = model(image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
```


```python
best_test_loss = None
for e in range(1, 150):
    train()
    test_loss, test_accuracy = test()
    print("\n[Epoch: %d] Test Loss:%5.5f Test Accuracy:%5.5f" % (e, test_loss, test_accuracy))
    # Save the model if the test_loss is the lowest
    if not best_test_loss or test_loss < best_test_loss:
        best_test_loss = test_loss
    else:
        break
print("\nFinal Results\n-------------\n""Loss:", best_test_loss, "Test Accuracy: ", test_accuracy)

    Final Results
    -------------
    Loss: 1.60090330081245 Test Accuracy:  0.8987
```
