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

def get_args():
    parser = argparse.ArgumentParser(description="Train CNN Model")
    parser.add_argument('--epochs', '-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', '-l', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help="Optimizer's momentum")
    parser.add_argument("--continue-training", "-t", type=bool, default=False,
                        help="continue training from  the last checkpoint or not")
    parser.add_argument('--checkpoint-dir', '-c', type=str, default='checkpoint', help='place to save checkpoint of model')
    parser.add_argument('--tensorboard', '-d', type=str, default='dashboad', help='place to save dashboard of model')
    parser.add_argument('--image-size', '-i', type=int, default=380, help='size of image')
    parser.add_argument('--early-stopping', '-s', type=int, default=5, help='early stopping')
    return parser.parse_args()

    # print(parser.epochs)
    # print(parser.lr)
    # print(parser.batch_size)
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


def train(arg):

    # thiết lập gpu để train (nếu có gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Resize((arg.image_size, arg.image_size)),
    ])

    temp_val = '/kaggle/input/knee-dataset/dataset/val'
    temp_train = '/kaggle/input/knee-dataset/dataset/train'

    val = ImageFolder(root='dataset/val', transform=transform)
    train = ImageFolder(root='dataset/train', transform=transform)

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

    num_class = len(train.classes)
    print(train.classes)
# -----------------------------------------------------------------------------------------

    model = CNN(num_class=num_class)
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
    criterion = nn.CrossEntropyLoss()

    # đưa model vào gpu (nếu có)
    model.to(device)

    best_acc = -999
    best_epoch = 0

    if arg.continue_training:
        checkpoint = torch.load(os.path.join(arg.checkpoint_dir, "last.pt"))
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
    if os.path.exists(arg.checkpoint_dir):
        shutil.rmtree(arg.checkpoint_dir)
    os.mkdir(arg.checkpoint_dir)

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

        # khởi tạo thanh tiến trình
        progress_bar = tqdm(train_loader, colour='yellow')

        # training
        for iter, (images, labels) in enumerate(progress_bar):

            # đưa data vào gpu (nếu có)
            images = images.to(device)
            labels = labels.to(device)

            # foward
            pred = model(images)
            loss = criterion(pred, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Ghi nhận loss
            loss_recorded.append(loss.item())

            # trực quan hóa tiến trình
            progress_bar.set_description(f'Epoch {epoch + 1}/{arg.epochs}, Loss: {loss.item():.4f}')
            writer.add_scalar(tag='Train/Loss', scalar_value=np.mean(loss_recorded), global_step= epoch*len(train_loader) + iter)

        # Khởi tạo danh sách lưu trữ kết quả dự đoán của mô hình
        all_labels = [] # Danh sách nhãn ban đầu của data
        all_preds = [] # Danh sách output mà model dự đoán
        all_loss = [] # Danh sách loss ghi nhận khi validation

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
            avg_loss = np.mean(all_loss)
            cm = confusion_matrix(all_labels, all_preds)

            # Trực quan hóa tiến trình đánh giá mô hình
            plot_confusion_matrix(writer, cm, class_names=train.classes, epoch=epoch)
            writer.add_scalar(tag='avg_loss/val', scalar_value=avg_loss, global_step=epoch)
            writer.add_scalar(tag='acc/val', scalar_value=acc, global_step=epoch)

            # show kết quả
            print(f'acc: {acc:.4f}, avg loss: {avg_loss:.4f}')

            # Lưu những thứ cần thiết của model để tái sử dụng hoặc tái huấn luyện trong lần tiếp theo
            checkpoint = {
                'model_state': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                "best_epoch": best_epoch,
                "best_accuracy": best_acc
            }

            # Mô tả thêm:
            #   'model_state' là trọng số của mô hình, chúng ta không lưu toàn bộ
            #   kiến trúc của mô hình vì rất tốn bộ nhớ cũng như làm cho src code
            #   trở nên rập khuôn.

            # Trong qua trình lưu, chúng ta sẽ chọn ra model state tốt nhất
            if acc > best_acc:
                best_acc = acc
                best_epoch = checkpoint['epoch']
                torch.save(checkpoint, os.path.join(arg.checkpoint_dir, 'best.pt'))


        # Ngoài ra, chúng ta sẽ chon ra model state cuối cùng. state này cũng có thể là state tốt nhất hoặc không phải
        torch.save(checkpoint, os.path.join(arg.checkpoint_dir, 'last.pt'))


        # if epoch - best_epoch > arg.early_stopping:
        #     print('Stop training at epoch {}'.format(best_epoch))
        #     exit(0)


if __name__ == '__main__':
    arg = get_args()
    train(arg)

