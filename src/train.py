from model import CNN
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import shutil, os
import numpy as np
from tqdm.autonotebook import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def parse_arg():
    parse = argparse.ArgumentParser(description='Transfer learning')
    parse.add_argument('--epochs', '-e', type=int, default=100, help='number of time model training')
    parse.add_argument('--checkpoint-dir', '-d', type=str, default='../checkpoint/my_model', help='where to place checkpoint')
    parse.add_argument('--tensorboard', '-t', type=str, default='../dashboad', help='place to store the visualization')
    parse.add_argument('--image-size', '-i', type=int, default=380, help='place to store the visualization')
    parse.add_argument('--batch-size', '-b', type=int, default=8)
    parse.add_argument('--continue-training', '-c', type=bool, default=False)
    parse.add_argument('--momentum', '-m', type=float, default=0.9, help="Optimizer's momentum")
    parse.add_argument('--lr', '-l', type=float, default=1e-3)
    parse.add_argument('--early-stopping', '-s', type=int, default=5, help='early stopping')

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

    temp_val = '/kaggle/input/knee-dataset/dataset/test'
    temp_train = '/kaggle/input/knee-dataset/dataset/train'

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


def train(arg):

    # thiết lập gpu để train (nếu có gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Resize((arg.image_size, arg.image_size)),
    ])

    temp_val = '/kaggle/input/knee-dataset/dataset/val'
    temp_train = '/kaggle/input/knee-dataset/dataset/train'

    val = ImageFolder(root='../dataset/val', transform=transform)
    train = ImageFolder(root='../dataset/train', transform=transform)

    train_loader = DataLoader(
        dataset=train,
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val,
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=True
    )

    classes = train.classes
# -----------------------------------------------------------------------------------------

    model = CNN(num_class=len(classes))
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
    criterion = nn.CrossEntropyLoss()
    
    # đưa model vào gpu (nếu có)
    model.to(device)

    best_acc = -999
    best_epoch = 0

    model_checkpoint_path = arg.checkpoint_dir
    
    if arg.continue_training:
        checkpoint = torch.load(os.path.join(model_checkpoint_path, "last.pt"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_accuracy = checkpoint["best_accuracy"]
    else:
        start_epoch = 0
        best_accuracy = -1
        best_epoch = 0

    # Trong quá trình lưu model state, chúng ta cần chọn ra state cuối cùng
    # Để làm điều đó, chúng ta xóa đi và tạo lại đường dẫn - nơi lưu checkpoint
    # trong mỗi một iteration, đến khi nào chạm mốc iter/epoch cuối cùng thì dừng lại
    if os.path.exists(model_checkpoint_path):
        shutil.rmtree(model_checkpoint_path)
    os.mkdir(model_checkpoint_path)

    # Chúng ta sử dụng tensorboard để trực quan hóa tiến trình train
    # Khi đó, torch.utils.tensorboard.SummaryWriter sẽ là lựa chọn tối ưu
    # Khởi tạo nơi lưu trũ tensorboard
    if not os.path.isdir(arg.tensorboard):
        os.mkdir(arg.tensorboard)

    list_board = [f for f in os.listdir(arg.tensorboard)]
    if len(list_board) > 0:
        for board in list_board:
            os.remove(os.path.join(arg.tensorboard, board))
    writer = SummaryWriter(arg.tensorboard)

    for epoch in range(arg.epochs):

        # Chỉ định model ở trạng thái training.
        # Nghĩa là model được phép dropout và được phép tính gradient
        model.train()

        # Khởi tạo danh sách lưu trữ loss mỗi lần tính được.
        loss_recorded = []

        # Khởi tạo danh sách lưu trữ kết quả dự đoán của mô hình
        all_labels = [] # Danh sách nhãn ban đầu của data
        all_preds = [] # Danh sách output mà model dự đoán
        all_loss = [] # Danh sách loss ghi nhận khi validation
        
        # khởi tạo thanh tiến trình
        progress_bar = tqdm(train_loader, colour='yellow')

        # training
        for iter, (images, labels) in enumerate(progress_bar):

            # đưa data vào gpu (nếu có)
            images = images.to(device)
            labels = labels.to(device)

            # foward
            pred = model(images)
            max_pred = torch.argmax(pred, dim=1)
            loss = criterion(pred, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Ghi nhận loss
            loss_recorded.append(loss.item())
            all_labels.extend(labels.tolist())
            all_preds.extend(max_pred.tolist())
            
            # trực quan hóa tiến trình
            progress_bar.set_description(f'Epoch {epoch + 1}/{arg.epochs}, Training/Loss: {loss.item():.4f}')
            writer.add_scalar(tag='Train/Loss', scalar_value=np.mean(loss_recorded), global_step= epoch*len(train_loader) + iter)

        

        # numerical metrics
        acc = accuracy_score(all_labels, all_preds)
        pre = precision_score(all_labels, all_preds, average='micro')
        rec = recall_score(all_labels, all_preds, average='micro')
        f1 = f1_score(all_labels, all_preds, average='micro')

        print(
            f'Acc: {acc:.4f}, precision: {pre:.4f}, recall: {rec:.4f}, f1_score: {f1:.4f} avg_oss: {np.mean(loss_recorded):.4f}')
        
        # Chỉ định model ở trạng thái evaluation. Nghĩa là model không có dropout, không có gradien
        model.eval()

        # Khởi tạo thanh tiến trình
        progress_bar = tqdm(val_loader, colour='yellow')

        # validation
        with torch.no_grad():
            for iter,(images, labels) in enumerate(progress_bar):

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

            # Đánh giá mô hình
            acc = accuracy_score(all_labels, all_preds)
            pre = precision_score(all_labels, all_preds, average='micro')
            rec = recall_score(all_labels, all_preds, average='micro')
            f1 = f1_score(all_labels, all_preds, average='micro')
            avg_loss = np.mean(all_loss)

            print(f'Validation: \n Acc: {acc:.4f}, precision: {pre:.4f}, recall: {rec:.4f}, f1_score: {f1:.4f} avg_oss: {avg_loss:.4f}')
            writer.add_scalar(tag='avg_loss/val', scalar_value=avg_loss, global_step=epoch)

            metrics = {
                'best_accuracy' : -999,
                'best_f1' : -999,
                'best_precision' : -999,
                'best_recall' : -999,
                'best_epoch' : 0
            }

            metrics_evaluations(writer, optimizer, epoch, acc, metrics, 'accuracy', model,
                                model_checkpoint_path)
            metrics_evaluations(writer, optimizer, epoch, f1, metrics, 'f1', model, model_checkpoint_path)

            
            metrics_evaluations(writer, optimizer, epoch, pre, metrics, 'precision', model,
                                model_checkpoint_path)
            metrics_evaluations(writer, optimizer, epoch, rec, metrics, 'recall', model,
                                model_checkpoint_path)
            
            cm = confusion_matrix(all_labels, all_preds)
            plot_roc_curve(writer, all_labels, all_preds, class_names=classes, epoch=epoch)

            # Trực quan hóa tiến trình đánh giá mô hình
            plot_confusion_matrix(writer, cm, class_names=classes, epoch=epoch)
            writer.add_scalar(tag='avg_loss/val', scalar_value=avg_loss, global_step=epoch)
            writer.add_scalar(tag='acc/val', scalar_value=acc, global_step=epoch)


        # if epoch - best_epoch > arg.early_stopping:
        #     print('Stop training at epoch {}'.format(best_epoch))
        #     exit(0)


if __name__ == '__main__':
    arg = parse_arg()
    train(arg)
