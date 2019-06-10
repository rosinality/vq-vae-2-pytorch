import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler


def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)

        if args.hier == 'top':
            target = top
            out = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=420)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)

    if args.hier == 'top':
        model = PixelSNAIL([32, 32], 512, 256, 5, 4, 4, 256)

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            256,
            5,
            3,
            4,
            256,
            attention=False,
            n_cond_res_block=3,
            cond_res_channel=256,
        )

    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, scheduler, device)
        torch.save(
            model.module.state_dict(),
            f'checkpoint/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )
