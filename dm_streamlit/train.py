import os

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# 数据预处理
data_transforms = {
    'train': Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

data_dir = 'food101'  # 更改为你的Food-101数据集路径

image_datasets = {x: Food101(os.path.join(data_dir, x), transform=data_transforms[x])
                 for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
              for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders['train'].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['val'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)

        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return epoch_acc

from model1 import *
# 初始化设备、模型、损失函数、优化器等
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SENet(num_classes=101).to(device)  # 确保num_classes匹配数据集类别数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 主训练循环
num_epochs = 25
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    train(model, dataloaders, criterion, optimizer, device)
    acc = validate(model, dataloaders, criterion, device)

    # 模型保存逻辑可以根据验证准确率进行调整
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'Saved best model with accuracy: {best_acc:.4f}')