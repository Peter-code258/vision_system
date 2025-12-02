# vision_system/models/utils.py
import torch
import matplotlib.pyplot as plt

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='model.pth'):
    model.load_state_dict(torch.load(path))

def plot_image(image, label):
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.title(f"Label: {label}")
    plt.show()
