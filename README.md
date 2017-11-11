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
```

    
    [Epoch: 1] Test Loss:2.27352 Test Accuracy:0.44360
    
    [Epoch: 2] Test Loss:2.22371 Test Accuracy:0.45100
    
    [Epoch: 3] Test Loss:2.16380 Test Accuracy:0.49840
    
    [Epoch: 4] Test Loss:2.09973 Test Accuracy:0.51520
    
    [Epoch: 5] Test Loss:2.04782 Test Accuracy:0.56200
    
    [Epoch: 6] Test Loss:2.00434 Test Accuracy:0.60630
    
    [Epoch: 7] Test Loss:1.96735 Test Accuracy:0.62930
    
    [Epoch: 8] Test Loss:1.93913 Test Accuracy:0.64160
    
    [Epoch: 9] Test Loss:1.91655 Test Accuracy:0.65620
    
    [Epoch: 10] Test Loss:1.89545 Test Accuracy:0.68240
    
    [Epoch: 11] Test Loss:1.87484 Test Accuracy:0.70650
    
    [Epoch: 12] Test Loss:1.85802 Test Accuracy:0.71700
    
    [Epoch: 13] Test Loss:1.84345 Test Accuracy:0.72550
    
    [Epoch: 14] Test Loss:1.82930 Test Accuracy:0.74690
    
    [Epoch: 15] Test Loss:1.81557 Test Accuracy:0.77430
    
    [Epoch: 16] Test Loss:1.80372 Test Accuracy:0.78770
    
    [Epoch: 17] Test Loss:1.79372 Test Accuracy:0.79150
    
    [Epoch: 18] Test Loss:1.78501 Test Accuracy:0.79350
    
    [Epoch: 19] Test Loss:1.77731 Test Accuracy:0.79600
    
    [Epoch: 20] Test Loss:1.77043 Test Accuracy:0.79800
    
    [Epoch: 21] Test Loss:1.76424 Test Accuracy:0.79990
    
    [Epoch: 22] Test Loss:1.75864 Test Accuracy:0.80170
    
    [Epoch: 23] Test Loss:1.75355 Test Accuracy:0.80300
    
    [Epoch: 24] Test Loss:1.74890 Test Accuracy:0.80510
    
    [Epoch: 25] Test Loss:1.74463 Test Accuracy:0.80620
    
    [Epoch: 26] Test Loss:1.74069 Test Accuracy:0.80720
    
    [Epoch: 27] Test Loss:1.73705 Test Accuracy:0.80880
    
    [Epoch: 28] Test Loss:1.73367 Test Accuracy:0.80960
    
    [Epoch: 29] Test Loss:1.73052 Test Accuracy:0.81040
    
    [Epoch: 30] Test Loss:1.72757 Test Accuracy:0.81110
    
    [Epoch: 31] Test Loss:1.72482 Test Accuracy:0.81170
    
    [Epoch: 32] Test Loss:1.72223 Test Accuracy:0.81150
    
    [Epoch: 33] Test Loss:1.71979 Test Accuracy:0.81260
    
    [Epoch: 34] Test Loss:1.71750 Test Accuracy:0.81350
    
    [Epoch: 35] Test Loss:1.71532 Test Accuracy:0.81350
    
    [Epoch: 36] Test Loss:1.71326 Test Accuracy:0.81490
    
    [Epoch: 37] Test Loss:1.71131 Test Accuracy:0.81560
    
    [Epoch: 38] Test Loss:1.70945 Test Accuracy:0.81610
    
    [Epoch: 39] Test Loss:1.70768 Test Accuracy:0.81660
    
    [Epoch: 40] Test Loss:1.70599 Test Accuracy:0.81710
    
    [Epoch: 41] Test Loss:1.70437 Test Accuracy:0.81810
    
    [Epoch: 42] Test Loss:1.70282 Test Accuracy:0.81840
    
    [Epoch: 43] Test Loss:1.70134 Test Accuracy:0.81910
    
    [Epoch: 44] Test Loss:1.69992 Test Accuracy:0.81960
    
    [Epoch: 45] Test Loss:1.69854 Test Accuracy:0.82030
    
    [Epoch: 46] Test Loss:1.69722 Test Accuracy:0.82110
    
    [Epoch: 47] Test Loss:1.69594 Test Accuracy:0.82090
    
    [Epoch: 48] Test Loss:1.69470 Test Accuracy:0.82140
    
    [Epoch: 49] Test Loss:1.69350 Test Accuracy:0.82170
    
    [Epoch: 50] Test Loss:1.69233 Test Accuracy:0.82180
    
    [Epoch: 51] Test Loss:1.69119 Test Accuracy:0.82220
    
    [Epoch: 52] Test Loss:1.69007 Test Accuracy:0.82240
    
    [Epoch: 53] Test Loss:1.68897 Test Accuracy:0.82280
    
    [Epoch: 54] Test Loss:1.68787 Test Accuracy:0.82320
    
    [Epoch: 55] Test Loss:1.68678 Test Accuracy:0.82370
    
    [Epoch: 56] Test Loss:1.68567 Test Accuracy:0.82450
    
    [Epoch: 57] Test Loss:1.68453 Test Accuracy:0.82490
    
    [Epoch: 58] Test Loss:1.68333 Test Accuracy:0.82580
    
    [Epoch: 59] Test Loss:1.68204 Test Accuracy:0.82700
    
    [Epoch: 60] Test Loss:1.68060 Test Accuracy:0.82880
    
    [Epoch: 61] Test Loss:1.67894 Test Accuracy:0.83060
    
    [Epoch: 62] Test Loss:1.67696 Test Accuracy:0.83360
    
    [Epoch: 63] Test Loss:1.67463 Test Accuracy:0.83750
    
    [Epoch: 64] Test Loss:1.67199 Test Accuracy:0.84090
    
    [Epoch: 65] Test Loss:1.66929 Test Accuracy:0.84380
    
    [Epoch: 66] Test Loss:1.66677 Test Accuracy:0.84850
    
    [Epoch: 67] Test Loss:1.66456 Test Accuracy:0.85150
    
    [Epoch: 68] Test Loss:1.66259 Test Accuracy:0.85390
    
    [Epoch: 69] Test Loss:1.66077 Test Accuracy:0.85540
    
    [Epoch: 70] Test Loss:1.65905 Test Accuracy:0.85720
    
    [Epoch: 71] Test Loss:1.65739 Test Accuracy:0.85920
    
    [Epoch: 72] Test Loss:1.65576 Test Accuracy:0.86050
    
    [Epoch: 73] Test Loss:1.65416 Test Accuracy:0.86220
    
    [Epoch: 74] Test Loss:1.65260 Test Accuracy:0.86360
    
    [Epoch: 75] Test Loss:1.65106 Test Accuracy:0.86480
    
    [Epoch: 76] Test Loss:1.64955 Test Accuracy:0.86590
    
    [Epoch: 77] Test Loss:1.64807 Test Accuracy:0.86770
    
    [Epoch: 78] Test Loss:1.64661 Test Accuracy:0.86940
    
    [Epoch: 79] Test Loss:1.64518 Test Accuracy:0.87040
    
    [Epoch: 80] Test Loss:1.64379 Test Accuracy:0.87180
    
    [Epoch: 81] Test Loss:1.64242 Test Accuracy:0.87240
    
    [Epoch: 82] Test Loss:1.64109 Test Accuracy:0.87360
    
    [Epoch: 83] Test Loss:1.63980 Test Accuracy:0.87450
    
    [Epoch: 84] Test Loss:1.63853 Test Accuracy:0.87590
    
    [Epoch: 85] Test Loss:1.63731 Test Accuracy:0.87790
    
    [Epoch: 86] Test Loss:1.63612 Test Accuracy:0.87870
    
    [Epoch: 87] Test Loss:1.63496 Test Accuracy:0.87950
    
    [Epoch: 88] Test Loss:1.63384 Test Accuracy:0.88010
    
    [Epoch: 89] Test Loss:1.63276 Test Accuracy:0.88130
    
    [Epoch: 90] Test Loss:1.63171 Test Accuracy:0.88230
    
    [Epoch: 91] Test Loss:1.63070 Test Accuracy:0.88320
    
    [Epoch: 92] Test Loss:1.62971 Test Accuracy:0.88380
    
    [Epoch: 93] Test Loss:1.62877 Test Accuracy:0.88490
    
    [Epoch: 94] Test Loss:1.62785 Test Accuracy:0.88620
    
    [Epoch: 95] Test Loss:1.62696 Test Accuracy:0.88650
    
    [Epoch: 96] Test Loss:1.62610 Test Accuracy:0.88750
    
    [Epoch: 97] Test Loss:1.62527 Test Accuracy:0.88740
    
    [Epoch: 98] Test Loss:1.62447 Test Accuracy:0.88810
    
    [Epoch: 99] Test Loss:1.62369 Test Accuracy:0.88830
    
    [Epoch: 100] Test Loss:1.62294 Test Accuracy:0.88880
    
    [Epoch: 101] Test Loss:1.62220 Test Accuracy:0.88930
    
    [Epoch: 102] Test Loss:1.62149 Test Accuracy:0.88970
    
    [Epoch: 103] Test Loss:1.62080 Test Accuracy:0.88990
    
    [Epoch: 104] Test Loss:1.62013 Test Accuracy:0.89040
    
    [Epoch: 105] Test Loss:1.61948 Test Accuracy:0.89060
    
    [Epoch: 106] Test Loss:1.61885 Test Accuracy:0.89110
    
    [Epoch: 107] Test Loss:1.61823 Test Accuracy:0.89170
    
    [Epoch: 108] Test Loss:1.61763 Test Accuracy:0.89190
    
    [Epoch: 109] Test Loss:1.61704 Test Accuracy:0.89230
    
    [Epoch: 110] Test Loss:1.61647 Test Accuracy:0.89230
    
    [Epoch: 111] Test Loss:1.61591 Test Accuracy:0.89290
    
    [Epoch: 112] Test Loss:1.61536 Test Accuracy:0.89320
    
    [Epoch: 113] Test Loss:1.61483 Test Accuracy:0.89340
    
    [Epoch: 114] Test Loss:1.61430 Test Accuracy:0.89330
    
    [Epoch: 115] Test Loss:1.61379 Test Accuracy:0.89360
    
    [Epoch: 116] Test Loss:1.61329 Test Accuracy:0.89380
    
    [Epoch: 117] Test Loss:1.61280 Test Accuracy:0.89400
    
    [Epoch: 118] Test Loss:1.61232 Test Accuracy:0.89420
    
    [Epoch: 119] Test Loss:1.61185 Test Accuracy:0.89430
    
    [Epoch: 120] Test Loss:1.61139 Test Accuracy:0.89430
    
    [Epoch: 121] Test Loss:1.61094 Test Accuracy:0.89430
    
    [Epoch: 122] Test Loss:1.61049 Test Accuracy:0.89470
    
    [Epoch: 123] Test Loss:1.61006 Test Accuracy:0.89500
    
    [Epoch: 124] Test Loss:1.60963 Test Accuracy:0.89500
    
    [Epoch: 125] Test Loss:1.60921 Test Accuracy:0.89510
    
    [Epoch: 126] Test Loss:1.60880 Test Accuracy:0.89500
    
    [Epoch: 127] Test Loss:1.60839 Test Accuracy:0.89500
    
    [Epoch: 128] Test Loss:1.60799 Test Accuracy:0.89500
    
    [Epoch: 129] Test Loss:1.60760 Test Accuracy:0.89500
    
    [Epoch: 130] Test Loss:1.60721 Test Accuracy:0.89520
    
    [Epoch: 131] Test Loss:1.60683 Test Accuracy:0.89550
    
    [Epoch: 132] Test Loss:1.60646 Test Accuracy:0.89570
    
    [Epoch: 133] Test Loss:1.60609 Test Accuracy:0.89580
    
    [Epoch: 134] Test Loss:1.60573 Test Accuracy:0.89630
    
    [Epoch: 135] Test Loss:1.60538 Test Accuracy:0.89660
    
    [Epoch: 136] Test Loss:1.60503 Test Accuracy:0.89660
    
    [Epoch: 137] Test Loss:1.60468 Test Accuracy:0.89680
    
    [Epoch: 138] Test Loss:1.60434 Test Accuracy:0.89710
    
    [Epoch: 139] Test Loss:1.60401 Test Accuracy:0.89730
    
    [Epoch: 140] Test Loss:1.60368 Test Accuracy:0.89750
    
    [Epoch: 141] Test Loss:1.60335 Test Accuracy:0.89780
    
    [Epoch: 142] Test Loss:1.60303 Test Accuracy:0.89790
    
    [Epoch: 143] Test Loss:1.60271 Test Accuracy:0.89820
    
    [Epoch: 144] Test Loss:1.60240 Test Accuracy:0.89840
    
    [Epoch: 145] Test Loss:1.60209 Test Accuracy:0.89850
    
    [Epoch: 146] Test Loss:1.60179 Test Accuracy:0.89870
    
    [Epoch: 147] Test Loss:1.60149 Test Accuracy:0.89860
    
    [Epoch: 148] Test Loss:1.60119 Test Accuracy:0.89880
    
    [Epoch: 149] Test Loss:1.60090 Test Accuracy:0.89870
    
    Final Results
    -------------
    Loss: 1.60090330081245 Test Accuracy:  0.8987


