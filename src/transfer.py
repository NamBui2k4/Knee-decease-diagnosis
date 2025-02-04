import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import (Compose,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    ToTensor,
                                    Normalize,
                                    Resize,
                                    CenterCrop
                                    )
import matplotlib.pyplot as plt
import argparse
import shutil
import warnings

from torchvision.models import efficientnet_b2
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


def parse_arg():
    parse = argparse.ArgumentParser(description='Transfer learning')
    parse.add_argument('--epochs', '-e', type=int, default=100, help='number of time model training')
    parse.add_argument('--checkpoint-dir', '-d', type=str, default='../checkpoint', help='where to place checkpoint')
    parse.add_argument('--tensorboard', '-b', type=str, default='../dashboad', help='place to store the visualization')
    parse.add_argument('--lr', '-l', type=float, default=1e-3)

    args, unknown = parse.parse_known_args()
    return args


def get_mean_std(loader):
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean.tolist(), std.tolist()


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="viridis")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def preprocessing():
    data_transforms = {
        'train': Compose([
            RandomResizedCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.574, 0.574, 0.574], [0.169, 0.169, 0.169])
        ]),
        'val': Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize([0.574, 0.574, 0.574], [0.169, 0.169, 0.169])
        ]),
    }

    temp_val = '/kaggle/input/data-knee-original/dataset/test'
    temp_train = '/kaggle/input/data-knee-original/dataset/train'

    train_set = ImageFolder(root='../dataset/train', transform=data_transforms['train'])
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_set = ImageFolder(root='../dataset/val', transform=data_transforms['val'])
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    return train_loader, val_loader


def transfer_learning(model, model_name, criterion, optimizer, parse, train_loader, val_loader):
    writer = SummaryWriter(parse.tensorboard)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # '/kaggle/working/' + parse.checkpoint_dir + "/" + model_name
    model_checkpoint_path = '/kaggle/working/' + parse.checkpoint_dir + "/" + model_name

    os.makedirs(model_checkpoint_path, exist_ok=True)

    if not os.path.exists(model_checkpoint_path):
        os.mkdir(model_checkpoint_path)

    for epoch in range(parse.epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{parse.epochs}')

        best_acc = -999
        best_epoch = 0

        # training
        model.train()
        loss_recorded = []
        all_classes = []
        all_labels = []

        progress_bar = tqdm(train_loader, colour='yellow')
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # foward
            pred = model(images)
            classes = torch.argmax(pred, dim=1)
            loss = criterion(pred, labels)

            # backward
            optimizer.zero_grad()  # dat lai gradient = 0, tranh tich luy gradientz
            loss.backward()  # lan truyen nguoc, grad = 3.221241234
            optimizer.step()  # cap nhat trong so

            all_labels.extend(labels.tolist())
            all_classes.extend(classes.tolist())

            loss_recorded.append(loss.item())

            progress_bar.set_description(f'Train/Batch {iter}, Loss: {loss.item():.4f}')
            writer.add_scalar(tag='Train/Loss', scalar_value=np.mean(loss_recorded),
                              global_step=epoch * len(train_loader) + iter)

        acc = accuracy_score(all_labels, all_classes)
        print(f'Acc: {acc}, Loss: {np.mean(loss_recorded)}')

        model.eval()
        all_labels = []  # Danh sách nhãn ban đầu của data
        all_outputs = []  # Danh sách output mà model dự đoán
        all_loss = []  # Danh sách loss ghi nhận khi validation

        progress_bar = tqdm(val_loader, colour='yellow')

        with torch.no_grad():
            for iter, (images, labels) in enumerate(progress_bar):
                # đưa data vào gpu (nếu có)
                images = images.to(device)
                labels = labels.to(device)

                # foward
                output = model(images)

                # predict
                loss = criterion(output, labels)
                prediction = torch.argmax(output, dim=1)

                # Ghi nhận kết quả
                all_loss.append(loss.item())
                all_labels.extend(labels.tolist())
                all_outputs.extend(prediction.tolist())

                progress_bar.set_description(f'Validation')

            # Đánh giá tren tap validation
            acc = accuracy_score(all_labels, all_outputs)
            avg_loss = np.mean(all_loss)
            cm = confusion_matrix(all_labels, all_outputs)

            plot_confusion_matrix(writer, cm, class_names=classes, epoch=epoch)

            writer.add_scalar(tag='avg_loss/val', scalar_value=avg_loss, global_step=epoch)
            writer.add_scalar(tag='acc/val', scalar_value=acc, global_step=epoch)

            # show kết quả
            print(f'Acc: {acc:.4f}, avg loss: {avg_loss:.4f}')

            checkpoint = {
                'state': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'best_epoch': best_epoch,
                'best_accuracy': best_acc
            }

            if best_acc < acc:
                best_acc = acc
                best_epoc = epoch
                torch.save(checkpoint, model_checkpoint_path + '/' + 'best.pt')


if __name__ == '__main__':
    train_loader, val_loader = preprocessing()
    parse = parse_arg()
    classes = lasses = ['Không mắc bệnh (normal)', 'có dấu hiệu thoái hóa (doubtful)', 'thoái hóa nhẹ (mild)',
                        'thoái hóa vừa phải (moderate)', 'thoái hóa nghiêm trọng (severe)',
                        'thiếu xương (osteopenia)', 'loãng xương (Osteoporosis)']
    if os.path.exists(parse.tensorboard):
        shutil.rmtree(parse.tensorboard)
    os.mkdir(parse.tensorboard)

    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

    model.classifier[1] = nn.Linear(model.last_channel, out_features=7, bias=True)

    optimizer = optim.Adam(model.parameters(), lr=parse.lr)
    # criterion
    criterion = nn.CrossEntropyLoss()
    
    transfer_learning(model, 'mobilenet_v2', criterion, optimizer, parse, train_loader, val_loader)