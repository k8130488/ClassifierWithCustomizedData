from utils.models import ResNet50
import torch
from loadData import loadDataSet
from torch.utils.data import DataLoader
from utils.saveCategoryInfo import saveAllInfo
import numpy as np
from utils.creatFolder import createSavePath
import torch.nn.functional as F


def test(data_loader, network):
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


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # 建立類神經網路模型，並放置於 GPU 或 CPU 上
    model_contain = ResNet50(3).to(device)
    model_contain.load_state_dict(torch.load('./runs/ICM/train/exp1/weight/model.pth'))
    model = model_contain
    batch_size_test = 8
    image_set = "ICM"
    test_set = loadDataSet(f"dataset/{image_set}/test")
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size_test, shuffle=False, pin_memory=True, num_workers=4)
    acc, model_output = test(test_loader, model)
    label_map = test_loader.dataset.labelmap
    saveRoot = f"./runs/{image_set}/test/"
    savePath = createSavePath(saveRoot)
    colors = np.random.uniform(0, 255, size=(len(label_map), 3)) / 255
    saveAllInfo(model_output, label_map, savePath, colors, lw=2)
