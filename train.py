
from torch import optim
from torch import nn
from utils import data_loader
from torchvision import models
import torch

train_loader = data_loader(True)
cuda = torch.cuda.is_available()
if cuda:
    resnet50 = models.resnet50(num_classes=61).cuda()
else:
    resnet50 = models.resnet50(num_classes=61)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=resnet50.parameters(),lr=0.01,weight_decay=1e-4)
epochs = 60
average_loss_series = []


for epoch in range(epochs):

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.cuda() if cuda else inputs
        labels = labels.cuda() if cuda else labels
        optimizer.zero_grad()
        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99: #每100个batch打印一次训练状态
            average_loss = running_loss/100
            print("[{0},{1}] loss:  {2}".format(epoch+1, i+1, average_loss))
            average_loss_series.append(average_loss)
            running_loss = 0.0

state = {'net':resnet50.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
torch.save(state, 'model.ckpt')