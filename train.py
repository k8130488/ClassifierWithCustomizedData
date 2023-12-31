import torch
from utils.loadData import loadDataSet
from utils.models import ResNet50, AlexNet
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils.creatFolder import createSavePath
import time
import argparse


def plot_history(history, savePath):
    plt.plot(history['acc'], marker='.')
    plt.plot(history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(f"{savePath}/accuracy.png")
    plt.close()

    plt.plot(history['loss'], marker='.')
    plt.plot(history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(f"{savePath}/loss.png")
    plt.close()


def train(data_loader, network, loss_fn, epoch, optimizer, device):
    network.train()
    correct = 0
    train_loss = 0
    n_class = data_loader.dataset.num_classes()
    for data, target in tqdm(data_loader, desc=f"Epoch {epoch}"):
        x, y = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(x)
        loss = loss_fn(output, y)
        # loss = loss_fn(output, F.one_hot(y, num_classes=n_class))
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).sum()
        train_loss += loss.item()
    train_loss /= len(data_loader.dataset)
    acc = '{:.4f}'.format(correct / len(data_loader.dataset))
    return float(acc), float(round(train_loss, 4))


def test(data_loader, network, loss_fn, device):
    network.eval()
    test_loss = 0
    correct = 0
    n_class = data_loader.dataset.num_classes()
    with torch.no_grad():
        for data, target in data_loader:
            x, y = data.to(device), target.to(device)
            output = network(x)
            test_loss += loss_fn(output, y).item()
            # test_loss += loss_fn(output, F.one_hot(y, num_classes=n_class)).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
    test_loss /= len(data_loader.dataset)
    acc = '{:.4f}'.format(correct / len(data_loader.dataset))
    return float(acc), float(round(test_loss, 4))


def run(
        source="./dataset",
        image_set="dogs_vs_cats",
        epochs=50,
        batch_size=8,
        workers=4,
        image_size=None,
        model="resnet50",
        cudnn=True,
        early_stop=False,
        save=True
):
    n_epochs = epochs
    batch_size_train = batch_size
    batch_size_test = batch_size
    # learning_rate = 0.01
    # momentum = 0.5
    # log_interval = 10
    random_seed = 1
    torch.manual_seed(random_seed)
    is_earlystop = early_stop
    model_name = model
    saveRoot = f"./runs/{image_set}/train/"
    savePath = createSavePath(saveRoot)
    # model_save_path = f"models/{image_set}/{model_name}/focal"
    model_save_path = f"{savePath}/weight"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    train_set = loadDataSet(f"{source}/{image_set}/train", resiz=(image_size, image_size))
    valid_set = loadDataSet(f"{source}/{image_set}/train", resiz=(image_size, image_size))
    # test_set = loadDataSet(f"dataset/{image_set}/test")
    train_loader = loader = DataLoader(
        dataset=train_set, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=workers)
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=workers)
    train_amount = np.array(train_loader.dataset.lbls_num())
    train_alpha = (train_amount / np.sum(train_amount))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    if device == "cuda" and cudnn:
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False
    # 建立類神經網路模型，並放置於 GPU 或 CPU 上
    if model_name == "resnet50":
        model = ResNet50(len(train_loader.dataset.labelmap)).to(device)
    elif model_name == "alexnet":
        model = AlexNet(len(train_loader.dataset.labelmap)).to(device)
    else:
        model = ResNet50(len(train_loader.dataset.labelmap)).to(device)
    # 損失函數
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = FocalLoss(alpha=train_alpha)
    # 學習優化器
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    test(valid_loader, model, loss_fn, device)
    valid_acc_keep = 0
    valid_loss_temp = 0
    valid_loss_zero = 0
    history = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}
    total_time_temp = []
    for epoch in range(1, n_epochs + 1):
        start = time.time()
        train_acc, train_loss = train(train_loader, model, loss_fn, epoch, optimizer, device)
        end = time.time()
        total_time_temp.append(end - start)
        history["acc"].append(train_acc)
        history["loss"].append(train_loss)
        print(f"train acc {train_acc} train loss {train_loss}")
        valid_acc, valid_loss = test(valid_loader, model, loss_fn, device)
        if len(history["val_loss"]) != 0:
            diff = valid_loss - history["val_loss"][-1]
            if diff > 0:
                valid_loss_temp += 1
            elif diff == 0:
                valid_loss_zero += 1
            else:
                valid_loss_zero = 0
                valid_loss_temp = 0
        history["val_acc"].append(valid_acc)
        history["val_loss"].append(valid_loss)
        print(f"val acc {valid_acc} val loss {valid_loss}")
        if valid_acc > valid_acc_keep and save:
            torch.save(model.state_dict(), f'./{model_save_path}/best.pth')
            torch.save(optimizer.state_dict(), f'./{model_save_path}/optimizer.pth')
        if is_earlystop:
            if valid_loss_temp > 5 or valid_loss_zero > 5:
                print(f"Early stop at the {epoch}_th epochs.")
                break
    if save:
        plot_history(history, savePath)
        torch.save(model.state_dict(), f'./{model_save_path}/final.pth')
    print(
        f"Training {epoch} epochs cost {round(np.sum(total_time_temp), 2)} seconds, average cost {round(np.mean(total_time_temp), 2)} s/epoch")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./dataset", help="images dataset root")
    parser.add_argument('--image-set', type=str, default="dogs_vs_cats", help="image set name")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--cudnn', type=bool, default=True)
    parser.add_argument('--early-stop', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=True)
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
