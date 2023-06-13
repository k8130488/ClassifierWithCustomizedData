from utils.models import ResNet50, AlexNet
import torch
from utils.loadData import loadDataSet
from torch.utils.data import DataLoader
from utils.saveCategoryInfo import saveAllInfo
import numpy as np
from utils.creatFolder import createSavePath
import torch.nn.functional as F
import argparse


def test(data_loader, network, device):
    network.eval()
    correct = 0
    y_temp = []
    predict_temp = []
    out_temp = []
    n_class = data_loader.dataset.num_classes()
    with torch.no_grad():
        for data, target in data_loader:
            # print(target)
            x, y = data.to(device), target.to(device)
            a = F.one_hot(y, num_classes=n_class)
            # print(a)
            output = network(x)
            # print(output)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
            y_temp.append(y)
            predict_temp.append(pred)
            out_temp.append(output)
            # print(f"y: {y}\n predict: {pred}\n corr {pred.eq(y.data.view_as(pred)).sum()}")
    acc = '{:.2f}'.format(100. * correct / len(data_loader.dataset))
    return acc, {"GT": y_temp, "predict": predict_temp, "score": out_temp}


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./dataset", help="images dataset root")
    parser.add_argument('--image-set', type=str, default="dogs_vs_cats", help="image set name")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--model_weight', type=str, default="./runs/dogs_vs_cats/train/exp1/weight/best.pth")
    opt = parser.parse_args()
    return opt


def run(
        source="./dataset",
        image_set="dogs_vs_cats",
        batch_size=8,
        workers=4,
        image_size=None,
        model="resnet50",
        model_weight="./runs/dogs_vs_cats/train/exp1/weight/best.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # 建立類神經網路模型，並放置於 GPU 或 CPU 上
    model_name = model
    batch_size_test = batch_size
    test_set = loadDataSet(f"{source}/{image_set}/test", resiz=(image_size, image_size))
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=workers)
    label_map = test_loader.dataset.labelmap
    if model_name == "resnet50":
        model_contain = ResNet50(len(label_map)).to(device)
    elif model_name == "alexnet":
        model_contain = AlexNet(len(label_map)).to(device)
    else:
        model_contain = ResNet50(len(label_map)).to(device)
    model_contain.load_state_dict(torch.load(model_weight))
    model = model_contain
    acc, model_output = test(test_loader, model, device)
    saveRoot = f"./runs/{image_set}/test/"
    savePath = createSavePath(saveRoot)
    colors = np.random.uniform(0, 255, size=(len(label_map), 3)) / 255
    saveAllInfo(model_output, label_map, savePath, colors, lw=2)


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
