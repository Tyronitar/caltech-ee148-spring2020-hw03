import argparse
import os

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


import numpy as np
import seaborn as sn
import pandas as pd

from main import Net

def visualize_miss(model: Net, device, test_loader):
    model.eval()
    incorrect_examples = []
    incorrect_labels = []
    incorrect_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            idxs_mask = ((pred == target.view_as(pred))==False).view(-1)
            if idxs_mask.numel(): #if index masks is non-empty append the data value in incorrect examples
                incorrect_examples.extend(data[idxs_mask].squeeze().cpu().numpy())
                incorrect_labels.extend(target[idxs_mask].cpu().numpy())
                incorrect_pred.extend(pred[idxs_mask].squeeze().cpu().numpy())

    fig, axs = plt.subplots(3, 3, constrained_layout=True)
    for i in range(9):
        img = incorrect_examples[i]
        ax = axs[int(i / 3), i % 3]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img, cmap='gray')
        ax.set_title(f"E: {incorrect_labels[i]}, P: {incorrect_pred[i]}")
    fig.suptitle("Missed Test Examples")
    fig.savefig("missed_ex.png")


def visualize_kernels(model: Net):

    kernels = model.conv1_1.weight.detach()

    fig, axs = plt.subplots(3, 3, constrained_layout=True)
    for i in range(3):
        for j in range(3):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(kernels[i + j].cpu().squeeze(), cmap='gray')
    
    fig.savefig("kernels.png")


def plot_confusion_matrix(model, device, test_loader):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            y_pred.extend(pred.squeeze().cpu())
            y_true.extend(target.squeeze().cpu())

    cm = confusion_matrix(y_true, y_pred)
    classes = list(range(10))

    df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, np.newaxis], index = [i for i in classes],
                     columns = [i for i in classes])

    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, fmt=".3f")
    plt.savefig('confusion_matrix.png')


def kNN(data, center, k):
    # Find k nearest neighbors
    # cloud is 4d tensor
    # center is target tensor
    dist = torch.norm(data - center, dim=1)
    vals, ids = dist.topk(k + 1, largest=False)
    return vals[1:], ids[1:]


def feature_repr(model: Net, device, test_loader):
    feature_vectors = []
    labels = []
    images = []
    n = 5

    model.eval()
    
    def copy_data(m, i, o):
        feature_vectors.append(o.squeeze().cpu())

    h = model.fc3.register_forward_hook(copy_data)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model(data)
            labels.extend([x.item() for x in target.squeeze().cpu()])
            images.extend(data.squeeze().cpu().numpy())
    
    h.remove()
    feature_vectors = torch.cat(feature_vectors, 0)

    fig = plt.figure()
    axs1 = fig.subplots(n, 1, gridspec_kw=dict(left=0.05, right=0.12, wspace=0.4))
    axs2 = fig.subplots(n, 8, gridspec_kw=dict(left=0.2, right=0.95, wspace=0.4))

    ids = torch.randint(0, len(feature_vectors), size=(n,))

    for i, id in enumerate(ids):
        axs1[i].set_xticks([])
        axs1[i].set_yticks([])
        axs1[i].imshow(images[id], cmap='gray')

        nn = kNN(feature_vectors, feature_vectors[id], 8)
        for j, n_id in enumerate(nn[1]):
            ax = axs2[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(images[n_id], cmap='gray')
    

    fig.savefig("nn.png")

    embedding = TSNE(n_components=2, learning_rate='auto',
                     init='random', verbose=1).fit_transform(feature_vectors)

    plt.figure(figsize=(16,10))
    sn.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=labels,
        palette=sn.color_palette("hls", 10),
        legend="full",
        alpha=0.3
    )
    plt.savefig('tsne.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--load-model', type=str, default='mnist_model.pt',
                        help='model file path')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    assert os.path.exists(args.load_model)

    # Set the test model
    model = Net().to(device)
    model.load_state_dict(torch.load(args.load_model))

    test_dataset = datasets.MNIST('../data', train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=True, **kwargs)

    visualize_miss(model, device, test_loader)
    visualize_kernels(model)
    plot_confusion_matrix(model, device, test_loader)
    feature_repr(model, device, test_loader)