from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, \
    precision_recall_curve, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def getPrecision_Recall_F1score(Y_true, Y_score, target_names):
    precision = dict()
    recall = dict()
    average_precision = dict()
    threshold = dict()
    f1_score = dict()
    for i, target_name in enumerate(target_names):
        # print(i)
        precision[target_name], recall[target_name], threshold[target_name] = precision_recall_curve(Y_true[:, i], Y_score[:, i])
        average_precision[target_name] = average_precision_score(Y_true[:, i], Y_score[:, i])
        f1_score[target_name] = 2 * ((precision[target_name] * recall[target_name]) / (precision[target_name] + recall[target_name]))

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], threshold["micro"] = precision_recall_curve(Y_true.ravel(), Y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_true, Y_score, average="micro")
    f1_score["micro"] = 2 * ((precision["micro"] * recall["micro"]) / (precision["micro"] + recall["micro"]))
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #       .format(average_precision["micro"]))
    return precision, recall, average_precision, f1_score, threshold


def getRocAuc(Y_true, Y_score, target_names):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, target_name in enumerate(target_names):
        fpr[target_name], tpr[target_name], _ = roc_curve(Y_true[:, i], Y_score[:, i])
        roc_auc[target_name] = auc(fpr[target_name], tpr[target_name])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_true.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[target_name] for target_name in target_names]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i, target_name in enumerate(target_names):
        mean_tpr += np.interp(all_fpr, fpr[target_name], tpr[target_name])

    # Finally average it and compute AUC
    mean_tpr /= len(target_names)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def saveROCCurve(filePath, fpr, tpr, roc_auc, target_names, colors, lw=2):
    plt.figure(figsize=(7, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(target_names, colors):
        plt.plot(fpr[i], tpr[i], color=tuple(color), lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc=(1, .7))
    fig.savefig(f"{filePath}/ROC_Curve.png", bbox_inches='tight')


def saveF1Curve(filePath, f1_score, threshold, average_precision, target_names, colors):
    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
    l, = plt.plot(np.r_[threshold["micro"], 1], f1_score["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average F1-score = {0:0.2f}'
                  ''.format(average_precision["micro"]))
    for i, color in zip(target_names, colors):
        l, = plt.plot(np.r_[threshold[i], 1], f1_score[i], color=tuple(color), lw=2)
        lines.append(l)
        labels.append('F1-score for class {0} ({1:0.2f})'
                      ''.format(i, average_precision[i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('threshold')
    plt.ylabel('f1-score')
    plt.title('Extension of F1-score curve')
    plt.legend(lines, labels, loc=(1, .7), prop=dict(size=14))
    fig.savefig(f"{filePath}/F1_Curve.png", bbox_inches='tight')


def savePRCurve(filePath, precision, recall, average_precision, target_names, colors):
    plt.figure(figsize=(7, 8))
    # f, (ax1) = plt.subplots(figsize=(10, 8), nrows=1)
    lines = []
    labels = []
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    for i, color in zip(target_names, colors):
        l, = plt.plot(recall[i], precision[i], color=tuple(color), lw=2)
        # l, = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    # colors.insert(0, "gold")
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(1, .7), prop=dict(size=14))
    # plt.legend(lines, labels, loc=1, prop=dict(size=14), fontsize="xx-small")
    fig.savefig(f"{filePath}/PR_Curve.png", bbox_inches='tight')


def saveClassificationReport(filePath, A):
    file = open(f"{filePath}/category_report.txt", 'w')
    file.write(A)
    file.close()


def saveConfusionMatrix(filePath, C2):
    sns.set()
    sns.set_context({"figure.figsize": (8, 8)})
    f, (ax2) = plt.subplots(figsize=(10, 8), nrows=1)
    heatmap = sns.heatmap(np.array(C2), annot=True, fmt="d", linewidths=0.2, annot_kws={'size': 32, 'weight': 'bold'}, square=True)
    ax2.set_title('sns_heatmap_confusion_matrix')
    ax2.set_xlabel('Pred')
    ax2.set_ylabel('True')
    f = heatmap.get_figure()
    f.savefig(f"{filePath}/confusion_matrix.png", bbox_inches='tight')


def twoLabel(y_temp, classes):
    temp = []
    first = classes[0]
    for i in y_temp:
        if i == first:
            temp.append([1, 0])
        else:
            temp.append([0, 1])
    return np.array(temp)


def saveAllInfo(model_output, label_map, savePath, colors, lw=2):
    y_temp = model_output["GT"]
    predict_temp = model_output["predict"]
    out_temp = model_output["score"]
    target_names = [label_map[x] for x in label_map]
    classes = list(np.arange(len(target_names)))
    y_true = [y.item() for temp in y_temp for y in temp]
    y_predict = [y.item() for temp in predict_temp for y in temp]
    y_score = [y.cpu().numpy() for temp in out_temp for y in temp]
    C2 = confusion_matrix(y_true, y_predict, labels=classes)
    A = classification_report(y_true, y_predict, target_names=target_names)
    if len(classes) == 2:
        Y_test = twoLabel(y_true, classes)
    elif len(classes) > 2:
        Y_test = label_binarize(y_true, classes=classes)
    else:
        print("Only one class exist.")
    Y_score = np.array(y_score)
    precision, recall, average_precision, f1_score, threshold = getPrecision_Recall_F1score(Y_test, Y_score,
                                                                                            target_names)
    fpr, tpr, roc_auc = getRocAuc(Y_test, Y_score, target_names)
    saveClassificationReport(savePath, A)
    print("Saving category report successfully")
    saveConfusionMatrix(savePath, C2)
    print("Saving confusion matrix successfully")
    savePRCurve(savePath, precision, recall, average_precision, target_names, colors)
    print("Saving PRCurve figure successfully")
    saveF1Curve(savePath, f1_score, threshold, average_precision, target_names, colors)
    print("Saving F1Curve figure successfully")
    saveROCCurve(savePath, fpr, tpr, roc_auc, target_names, colors, lw)
    print("Saving ROCCurve figure successfully")
