# vision_system/models/evaluate.py
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import SimpleCNN
from sklearn.metrics import accuracy_score


def evaluate(model, val_loader):
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def main():
    # 数据集加载
    data_dir = 'path/to/images'
    labels_file = 'path/to/labels.csv'

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(data_dir, labels_file, transform)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = SimpleCNN(num_classes=2).cuda()
    model.load_state_dict(torch.load('model.pth'))

    evaluate(model, val_loader)


if __name__ == "__main__":
    main()
