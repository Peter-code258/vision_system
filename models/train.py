# vision_system/models/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import SimpleCNN
import os


def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # 使用 GPU

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')


def main():
    # 参数
    data_dir = 'path/to/images'
    labels_file = 'path/to/labels.csv'

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(data_dir, labels_file, transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleCNN(num_classes=2).cuda()  # 假设有 2 类
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer)


if __name__ == "__main__":
    main()
