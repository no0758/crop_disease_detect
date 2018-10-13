
from torch import optim
from torch import nn
from utils import data_loader
from torchvision import models

train_loader = data_loader(True)
resnet50 = models.resnet50(num_classes=61)
criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(params=resnet50.parameters(),lr=0.001)

epochs = 2
average_loss_series = []
for epoch in range(epochs):

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9: #每10个batch打印一次训练状态
            average_loss = running_loss/10
            print("[{0},{1}] loss:  {2}".format(epoch+1, i+1, average_loss))
            average_loss_series.append(average_loss)
            running_loss = 0.0