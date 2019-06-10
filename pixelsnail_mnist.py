import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from pixelsnail import PixelSNAIL


def train(epoch, loader, model, optimizer, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out = model(img)
        loss = criterion(out, img)
        loss.backward()

        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == img).float()
        accuracy = correct.sum() / img.numel()

        loader.set_description(
            (f'epoch: {epoch + 1}; loss: {loss.item():.5f}; ' f'acc: {accuracy:.5f}')
        )


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    device = 'cuda'
    epoch = 10

    dataset = datasets.MNIST('.', transform=PixelTransform(), download=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = PixelSNAIL([28, 28], 256, 128, 5, 2, 4, 128)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(10):
        train(i, loader, model, optimizer, device)
        torch.save(model.state_dict(), f'checkpoint/mnist_{str(i + 1).zfill(3)}.pt')
