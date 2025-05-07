import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import json
from sklearn.preprocessing import label_binarize
import matplotlib as mpl


from pylab import *
ftsz = 8
mpl.rcParams['font.size'] = ftsz
mpl.rcParams['xtick.labelsize'] = ftsz # Set the fontsize of the horizontal axis label to 8
mpl.rcParams['ytick.labelsize'] = ftsz # Set the fontsize of the vertical axis label to 8

# Draw ROC curve
def plot_roc_mul(true_path1, pred_path1,true_path2, pred_path2,true_path3, pred_path3,true_path4, pred_path4,true_path5, pred_path5):
    with open(true_path1, 'r') as file:
        y_true1 = json.load(file)
    with open(pred_path1, 'r') as file:
        y_pred1 = json.load(file)
    
    with open(true_path2, 'r') as file:
        y_true2 = json.load(file)
    with open(pred_path2, 'r') as file:
        y_pred2 = json.load(file)
    
    with open(true_path3, 'r') as file:
        y_true3 = json.load(file)
    with open(pred_path3, 'r') as file:
        y_pred3 = json.load(file)
    
    with open(true_path4, 'r') as file:
        y_true4 = json.load(file)
    with open(pred_path4, 'r') as file:
        y_pred4 = json.load(file)
    
    with open(true_path5, 'r') as file:
        y_true5 = json.load(file)
    with open(pred_path5, 'r') as file:
        y_pred5 = json.load(file)
    
    # One-hot encoding of the actual tags
    y_true_bin1 = label_binarize(y_true1, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    n_classes1 = y_true_bin1.shape[1]
    y_pred1 = np.array(y_pred1)

    y_true_bin2 = label_binarize(y_true2, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    n_classes2 = y_true_bin2.shape[1]
    y_pred2 = np.array(y_pred2)

    y_true_bin3 = label_binarize(y_true3, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    n_classes3 = y_true_bin3.shape[1] 
    y_pred3 = np.array(y_pred3)

    y_true_bin4 = label_binarize(y_true4, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    n_classes4 = y_true_bin4.shape[1] 
    y_pred4 = np.array(y_pred4)

    y_true_bin5 = label_binarize(y_true5, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    n_classes5 = y_true_bin5.shape[1]
    y_pred5 = np.array(y_pred5)

    fpr1,tpr1 = dict(),dict()
    fpr2, tpr2 = dict(), dict()
    fpr3, tpr3 = dict(), dict()
    fpr4, tpr4 = dict(), dict()
    fpr5, tpr5 = dict(), dict()
    roc_auc1 = dict()
    roc_auc2 = dict()
    roc_auc3 = dict()
    roc_auc4 = dict()
    roc_auc5 = dict()

    # Calculate FPR, TPR, and AUC for each category
    for i in range(n_classes1):
        fpr1[i], tpr1[i], _ = roc_curve(y_true_bin1[:, i], y_pred1[:, i])
        roc_auc1[i] = auc(fpr1[i], tpr1[i])

    for i in range(n_classes1):
        fpr2[i], tpr2[i], _ = roc_curve(y_true_bin2[:, i], y_pred2[:, i])
        roc_auc2[i] = auc(fpr2[i], tpr2[i])

    for i in range(n_classes3):
        fpr3[i], tpr3[i], _ = roc_curve(y_true_bin3[:, i], y_pred3[:, i])
        roc_auc3[i] = auc(fpr3[i], tpr3[i])

    for i in range(n_classes4):
        fpr4[i], tpr4[i], _ = roc_curve(y_true_bin4[:, i], y_pred4[:, i])
        roc_auc4[i] = auc(fpr4[i], tpr4[i])

    for i in range(n_classes5):
        fpr5[i], tpr5[i], _ = roc_curve(y_true_bin5[:, i], y_pred5[:, i])
        roc_auc5[i] = auc(fpr5[i], tpr5[i])

    # Plot the ROC curve for each category
    c = 7
    plt.title('ROC')
    plt.plot(fpr1[c], tpr1[c], 'b', label='CropCapsNet AUC = %0.4f' % roc_auc1[c])
    plt.plot(fpr2[c], tpr2[c], 'r', label='ConvNeXt AUC = %0.4f' % (roc_auc2[c]))
    plt.plot(fpr3[c], tpr3[c], 'g', label='Efficient AUC = %0.4f' % roc_auc3[c])
    plt.plot(fpr4[c], tpr4[c], 'm', label='RegNet AUC = %0.4f' % roc_auc4[c])
    plt.plot(fpr5[c], tpr5[c], 'y', label='ResNeSt50 AUC = %0.4f' % roc_auc5[c])

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    # plt.savefig("1.png")


if __name__ == '__main__':

    root = "C:\\Users\\30534\\Desktop\\NEUROCOMPUTING\\实验数据\\PlantVillage_38\\"
    path1 = root + "600\\"
    path2 = root + "convnext\\"
    path3 = root + "efficientnet\\"
    path4 = root + "regnet_x_16gf\\"
    path5 = root + "resnest50\\"

    # plot_roc_mul(path1+"true_label.json",path1+"pred_prob.json",
    #              path2+"true_label.json",path2+"pred_prob.json",
    #              path3+"true_label.json",path3+"pred_prob.json",
    #              path4+"true_label.json",path4+"pred_prob.json",
    #              path5+"true_label.json",path5+"pred_prob.json",
    #             )



# Define the accuracy curve drawing function
def matplot_acc(model1, model2, model3, model4, model5):
    #draw the accuracy curve
    plt.plot(model1, label='InceptionV4')
    plt.plot(model2, label='Inception_DA')
    plt.plot(model3, label='Conv_InceptDA')
    plt.plot(model4, label='CpnvInceptRes_DA')
    plt.plot(model5, label='DAMSLNet')
    # set the range of y-axis
    plt.ylim((0, 1))
    plt.legend(loc='best')
    #set the name of x-axis and y-axis
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title("Accuracy of different models")
    plt.show()
    plt.savefig("Accuracy of defferent models.jpg")



import matplotlib.pyplot as plt
import json
data_path_1 = ".\\best_model.json"
data_path_2 = ".\\best_model.json"
data_path_3 = ".\\best_model.json"
data_path_4 = ".\\best_model.json"
data_path_5 = ".\\best_model.json"



acc1= []
acc2= []
acc3= []
acc4= []
acc5= []


with open(data_path_1, 'r') as file:
    model1 = json.load(file)
    acc1.append(model1[3])  # Extract the fourth column of data
    acc1 = acc1[0]
    # print(acc1)
with open(data_path_2, 'r') as file:
    model2 = json.load(file)
    acc2.append(model2[3])  # Extract the fourth column of data
    acc2 = acc2[0]
with open(data_path_3, 'r') as file:
    model3 = json.load(file)
    acc3.append(model3[3])  # Extract the fourth column of data
    acc3 = acc3[0]
with open(data_path_4, 'r') as file:
    model4 = json.load(file)
    acc4.append(model4[3])  # Extract the fourth column of data
    acc4 = acc4[0]
with open(data_path_5, 'r') as file:
    model5 = json.load(file)
    acc5.append(model5[3])  # Extract the fourth column of data
    acc5 = acc5[0]


matplot_acc(acc1, acc2, acc3, acc4, acc5)

