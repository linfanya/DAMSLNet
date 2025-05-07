import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
import numpy as np

import json
import matplotlib as mpl

from pylab import *
ftsz = 8
mpl.rcParams['font.size'] = ftsz
mpl.rcParams['xtick.labelsize'] = ftsz # Set the fontsize of the horizontal axis label to 8
mpl.rcParams['ytick.labelsize'] = ftsz # Set the fontsize of the vertical axis label to 8

device = torch.device("cuda")


def plot_confusion(true_path, pred_path,classes):
    # Generate confusion matrix
    with open(true_path, 'r') as file:
        y_true = json.load(file)
    with open(pred_path, 'r') as file:
        y_pred = json.load(file)
    cm = confusion_matrix(y_true, y_pred)
    cm = normalize(cm, axis=1, norm='l1')
    TP = cm[0][0] 
    FP = cm[:, 0].sum() - cm[0, 0]
    TN = cm[1:, 1:].sum()
    FN = cm[0, :].sum() - cm[0, 0]
    print(cm)
    print('TP: ', TP)
    print('FP: ', FP)
    print('TN: ', TN)
    print('FN: ', FN)

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)


    # Draw a confusion matrix diagram
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    plt.show()
    plt.savefig("CIRDANet_confu.jpg")

# Draw ROC curve
def plot_roc(true_path, pred_path):
    with open(true_path, 'r') as file:
        y_true = json.load(file)
    with open(pred_path, 'r') as file:
        y_pred = json.load(file)

     # One hot encoding on actual labels
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    n_classes = y_true_bin.shape[1] #12分类
    y_pred = np.array(y_pred)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Obtain ROC value of 12 classes separately
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["avg"], tpr["avg"], _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
    roc_auc["avg"] = auc(fpr["avg"], tpr["avg"])

    c = 'avg'
    plt.title('ROC')
    plt.plot(fpr[c], tpr[c], 'b', label='AUC = %0.4f' % roc_auc[c])
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    plt.savefig("roc.jpg")


if __name__ == '__main__':

    classes = [0,1,2,3,4,5,6,7,8,9,10,11]
    plot_confusion("./true_label.json","./pred_label.json", classes)
    # plot_roc("./true_label.json","./pred_prob.json")

