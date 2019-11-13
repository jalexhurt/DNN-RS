import os
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import find_classes
from tqdm import trange

from models import load_model


def main():
    parser = ArgumentParser()
    parser.add_argument("--test_data_dir", default=None)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--network", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--cpu_count", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    args = vars(parser.parse_args())

    params = {}
    for p in args:
        if p is not None:
            params[p] = args[p]

    validate(**params)


def validate(test_data_dir="./images",
             filename="mode.pt",
             network="ResNet50",
             gpu=None,
             cpu_count=os.cpu_count(),
             batch_size=16):
    classes, _ = find_classes(test_data_dir)
    model = load_model(filename, network, len(classes))

    cuda_id = int(gpu.split(",")[0]) if gpu is not None else 0
    device = torch.device("cuda:{}".format(cuda_id) if gpu is not None else "cpu")
    model = model.to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = ImageFolder(
        test_data_dir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))
    criterion = torch.nn.CrossEntropyLoss()

    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=cpu_count, batch_size=batch_size)
    # switch to evaluate mode
    model.eval()

    # setup running values
    running_loss = 0.0
    running_corrects = 0
    loss = 0.
    acc = 0.

    y_pred = []
    y_true = []
    conf = []

    total_seen_samples = 0
    with torch.no_grad():
        with trange(len(val_loader), desc="Validating", ncols=80, postfix={"loss": 0, "acc": 0},
                    bar_format="{desc}: {percentage:3.1f}% {bar} {remaining} {n_fmt}/{total_fmt}{postfix}") as pbar:
            start_time = perf_counter()
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                batch_size = inputs.size(0)
                total_seen_samples += batch_size
                labels = labels.to(device)

                # compute output
                output = model(inputs)
                preds = torch.argmax(output, 1)
                loss = criterion(output, labels)

                y_pred += preds.cpu().numpy().tolist()
                y_true += labels.cpu().numpy().tolist()
                conf += output.cpu().numpy().tolist()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                loss = running_loss / (i + 1)
                acc = running_corrects.double() / total_seen_samples

                pbar.set_postfix({"loss": round(float(loss), 2), "acc": round(float(acc), 3)})
                pbar.update()

            end_time = perf_counter()

    print("Loss: {:.4f}, Acc: {:.4f}, Time: {:.4f}s".format(loss, acc, end_time - start_time))

    return np.array(y_pred), np.array(y_true), np.array(conf)


if __name__ == '__main__':
    main()
