import datetime
import os
from argparse import ArgumentParser
from time import perf_counter

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import trange

from models import get_model


def main():
    parser = ArgumentParser()
    parser.add_argument("--train_data_dir", default=None)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--workers", default=None, type=int)
    parser.add_argument("--network", default=None)
    parser.add_argument("--initial_learning_rate", default=None, type=float)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--stat_filename", default=None)
    parser.add_argument("--output_filename", default=None)
    args = vars(parser.parse_args())

    params = {k: args[k] for k in args if args[k] is not None}

    print(params)

    train(**params)


def train(
        train_data_dir="./images",
        batch_size=16,
        workers=os.cpu_count(),
        network="ResNet50",
        initial_learning_rate=1e-4,
        gpu=None,
        epochs=1,
        stat_filename="train_stats.csv",
        output_filename="model.pt"
):
    use_gpu = (gpu is not None)

    ############
    # Create Data Loader
    ############
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = ImageFolder(
        train_data_dir,
        transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))

    # extract number of classes
    num_classes = len(set(dataset.classes))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # instantiate model
    model = get_model(network, num_classes)

    # setup loss function
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)

    #########
    # Device Setup
    #########
    cuda_id = int(gpu.split(",")[0]) if gpu is not None else 0
    gpu_device = torch.device("cuda:{}".format(cuda_id)) if gpu is not None else "cpu"
    if use_gpu:
        model = model.cuda(gpu_device)
        
    train_stats = pd.DataFrame(columns=["time", "loss", "acc"])

    for i in range(epochs):
        # train for one epoch
        model, epoch_loss, epoch_acc, epoch_time = epoch(dataloader, model, criterion, optimizer, i, gpu_device)

    # append to dataframe
    train_stats = train_stats.append({
        "time": epoch_time,
        "loss": epoch_loss.item() if isinstance(epoch_loss, torch.Tensor) else epoch_loss,
        "acc": epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc
    }, ignore_index=True)

    # save output file
    train_stats.to_csv(stat_filename, index_label="epoch")

    # final save
    save_checkpoint(model, output_filename)


def epoch(dataloader, model, loss_function, optimizer, epoch, gpu=None):
    """
    Train the model given the params
    :param model: the model to train
    :param dataloader: the data loader
    :param optimizer: the optimizer
    :param loss_function: the loss function to use
    :param epoch: which epoch this is
    :param args: which device to use
    :return: trained model
    """
    device = gpu if gpu is not None else "cpu"
    ##############
    # Perform Training
    ##############

    # set to training mode
    model.train()

    # setup running values
    running_loss = 0.0
    running_corrects = 0

    total_seen_samples = 0
    # Iterate over data.
    with trange(len(dataloader), total=len(dataloader), ncols=80, postfix={"loss": 0, "acc": 0},
                bar_format="{desc}: {percentage:3.1f}% {bar} {remaining} {n_fmt}/{total_fmt}{postfix}",
                desc="Epoch {}".format(epoch)) as pbar:
        start = perf_counter()
        for i, (inputs, labels) in enumerate(dataloader):
            batch_size = inputs.size(0)
            total_seen_samples += batch_size
            if device != "cpu":
                inputs = inputs.cuda(gpu, non_blocking=True)
                labels = labels.cuda(gpu, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_function(outputs.float(), labels.long())

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.long().data)
            loss = running_loss / (i + 1)
            acc = running_corrects.double() / total_seen_samples
            pbar.set_postfix({"loss": round(float(loss), 2), "acc": round(float(acc), 3)})
            pbar.update()

        end = perf_counter()

    epoch_avg_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / total_seen_samples

    print('Loss: {:.4f} \nAcc: {:.4f}'.format(
        epoch_avg_loss, epoch_acc))
    print("\t{} | {}".format(running_corrects.item(), total_seen_samples))
    print("Time: {}".format(str(datetime.timedelta(seconds=end - start))))

    return model, epoch_avg_loss, epoch_acc.item(), end - start


def save_checkpoint(model, filename='checkpoint.pth.tar'):
    """
    Save the model to the file
    :param model: model to wave
    :param filename: filename
    :return:
    """
    # determine what to save
    object = model
    # move module.conv1 --> conv1
    # and module.1.conv1 --> conv1
    path = ["module", "1"]
    # for each object in path
    for p in path:
        # see if this exists
        if hasattr(object, p):
            # move down a level
            object = getattr(object, p)
        # the path doesnt exist, just save here
        else:
            break
    # grab correct state dict
    state_dict = object.state_dict()
    # save it
    torch.save(state_dict, filename)


if __name__ == '__main__':
    main()
