import os.path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights, vgg11 , VGG11_Weights, resnet50, ResNet50_Weights
from torchvision.transforms import (Compose,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    ToTensor,
                                    Normalize,
                                    Resize,
                                    CenterCrop
                                    )
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse
import shutil
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def parse_arg():
    parse = argparse.ArgumentParser(description='Transfer learning')
    parse.add_argument('--epochs', '-e', type=int, default=10, help='number of time model training')
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


def plot_roc_curve(writer, y_true, y_score, class_names, epoch):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colours = ['blue', 'red', 'green', 'purple', 'orange', 'yellow', 'pink']
    # Chạy vòng lặp qua từng class và tính ROC cho từng class
    for i, class_name in enumerate(class_names):
        # Tạo nhãn nhị phân cho class hiện tại: 1 là class i, 0 là các class khác
        binary_labels = [1 if label == i else 0 for label in y_true]

        # Chuyển dự đoán thành xác suất hoặc giá trị dự đoán tương ứng cho class i
        binary_scores = [1 if score == i else 0 for score in y_score]

        # Tính ROC cho class i
        fpr[i], tpr[i], _ = roc_curve(binary_labels, binary_scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve cho từng class
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(len(class_names)), colours):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    writer.add_figure('roc_curve', plt.gcf(), epoch)
    plt.show()
    plt.close()


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

def metrics_evaluations(writer, optimizer, epoch, score, metrics, name, model,
                        model_checkpoint_path):
    writer.add_scalar(tag=f'{name}/val', scalar_value=score, global_step=epoch)

    # show kết quả

    checkpoint = {
        'state': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'best_epoch': metrics['best_epoch'],
        f'best_{name}': metrics[f'best_{name}']
    }    
    
    print('current best', metrics[f'best_{name}'])
    print('score', score)
    
    if metrics[f'best_{name}'] < score:
        metrics[f'best_{name}'] = score
        metrics[f'best_epoch'] = epoch
        torch.save(checkpoint, model_checkpoint_path + '/' + f'best_{name}.pt')




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

    # temp_val = '/kaggle/input/knee-dataset/dataset/test'
    # temp_train = '/kaggle/input/knee-dataset/dataset/train'

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


def transfer_learning(model, model_name, classes, criterion, optimizer, parse, train_loader, val_loader):
    tensorboard_path = parse.tensorboard + '/' + model_name
    os.makedirs(tensorboard_path, exist_ok=True)
    
    writer = SummaryWriter(tensorboard_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # create model checkpoint path is not exist

    model_checkpoint_path = os.path.join(parse.checkpoint_dir, model_name)
    os.makedirs(model_checkpoint_path, exist_ok=True)
    
    metrics = {
                'best_accuracy' : -999,
                'best_f1' : -999,
                'best_precision' : -999,
                'best_recall' : -999,
                'best_epoch' : 0
    }
    
    for epoch in range(parse.epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{parse.epochs}')

        # training
        model.train()
        loss_recorded = []
        all_pred = []
        all_labels = []

        progress_bar = tqdm(train_loader, colour='yellow')
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # foward
            pred = model(images)
            max_pred = torch.argmax(pred, dim=1)
            loss = criterion(pred, labels)

            # backward
            optimizer.zero_grad()  # dat lai gradient = 0, tranh tich luy gradientz
            loss.backward()  # lan truyen nguoc, grad = 3.221241234
            optimizer.step()  # cap nhat trong so

            all_labels.extend(labels.tolist())
            all_pred.extend(max_pred.tolist())

            loss_recorded.append(loss.item())

            progress_bar.set_description(f'Train/Batch {iter}, Loss: {loss.item():.4f}')
            writer.add_scalar(tag='Train/Loss', scalar_value=np.mean(loss_recorded),
                              global_step=epoch * len(train_loader) + iter)

        # numerical metrics
        acc = accuracy_score(all_labels, all_pred)
        pre = precision_score(all_labels, all_pred, average='micro')
        rec = recall_score(all_labels, all_pred, average='micro')
        f1 = f1_score(all_labels, all_pred,average='micro')

        print(f'Acc: {acc:.4f}, precision: {pre:.4f}, recall: {rec:.4f}, f1_score: {f1:.4f} avg_oss: {np.mean(loss_recorded):.4f}')

        model.eval()
        all_labels = []  # Danh sách nhãn ban đầu của data
        all_preds = []
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
                all_preds.extend(prediction.tolist())

                progress_bar.set_description(f'Validation')

            # numerical etrics
            acc = accuracy_score(all_labels, all_preds)
            pre = precision_score(all_labels, all_preds, average='micro')
            rec = recall_score(all_labels, all_preds, average='micro')
            f1 = f1_score(all_labels, all_preds,average='micro')
            avg_loss = np.mean(all_loss)

            print(f'Acc: {acc:.4f}, precision: {pre:.4f}, recall: {rec:.4f}, f1_score: {f1:.4f} avg_oss: {avg_loss:.4f}')
            writer.add_scalar(tag='avg_loss/val', scalar_value=avg_loss, global_step=epoch)

            # metrics evaluations
            metrics_evaluations(writer, optimizer, epoch, acc, metrics, 'accuracy', model,
                                model_checkpoint_path)
            metrics_evaluations(writer, optimizer, epoch, f1, metrics, 'f1', model, model_checkpoint_path)

            
            metrics_evaluations(writer, optimizer, epoch, pre, metrics, 'precision', model,
                                model_checkpoint_path)
            metrics_evaluations(writer, optimizer, epoch, rec, metrics, 'recall', model,
                                model_checkpoint_path)

            # confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plot_confusion_matrix(writer, cm, classes, epoch)

            # ROC curve
            plot_roc_curve(writer, all_labels, all_preds, classes, epoch)


if __name__ == '__main__':
    train_loader, val_loader = preprocessing()
    parse = parse_arg()
    classes = ['Không mắc bệnh (normal)', 'có dấu hiệu thoái hóa (doubtful)',
                'thoái hóa nhẹ (mild)', 'thoái hóa vừa phải (moderate)',
                'thoái hóa nghiêm trọng (severe)',
                'thiếu xương (osteopenia)', 'loãng xương (Osteoporosis)']

    if os.path.exists(parse.tensorboard):
        shutil.rmtree(parse.tensorboard)
    os.mkdir(parse.tensorboard)

    # criterion
    criterion = nn.CrossEntropyLoss()
    
    
    # --------------------------------- ResNet18 ---------------------------------
    resnet_18 = resnet18(weights=ResNet18_Weights)
    resnet_18.fc = nn.Linear(in_features=512, out_features=len(classes), bias=True)
    optimizer = optim.Adam(resnet_18.parameters(), lr=parse.lr)
    transfer_learning(resnet_18,
                        'resnet_18',
                        classes,
                        criterion,
                        optimizer,
                        parse,
                        train_loader,
                        val_loader
                        )
    
#    # --------------------------------- VGG11 ---------------------------------
#     vgg_11 = vgg11(weights=VGG11_Weights)
#     vgg_11.classifier[6] = nn.Linear(in_features=4096, out_features=len(classes), bias=True)
#     optimizer = optim.Adam(vgg_11.parameters(), lr=parse.lr)
#     transfer_learning(resnet_18,
#                         'vgg11',
#                         classes,
#                         criterion,
#                         optimizer,
#                         parse,
#                         train_loader,
#                         val_loader
#                         ) 
#     # --------------------------------- mobilenet_v2 ---------------------------------
    
#     mobilenet = mobilenet_v2(weights=MobileNetV2_Weights)
#     mobilenet.classifier[1] = nn.Linear(in_features=1280, out_features=len(classes), bias=True)
#     optimizer = optim.Adam(vgg_11.parameters(), lr=parse.lr)
#     transfer_learning(mobilenet,
#                         'mobilenet_v2',
#                         classes,
#                         criterion,
#                         optimizer,
#                         parse,
#                         train_loader,
#                         val_loader
#                         ) 