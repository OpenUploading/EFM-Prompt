import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix,cohen_kappa_score,f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import torch.nn.functional as F

# 解决中文显示问题
# 运行配置参数中的字体（font）为黑体（SimHei）
plt.rcParams['font.sans-serif'] = ['simHei']
# # 运行配置参数总的轴（axes）正常显示正负号（minus）
plt.rcParams['axes.unicode_minus'] = False


# 定义画图函数
# 错误率
def plot_loss(train_loss, val_loss,fname):
    # 参数label = ''传入字符串类型的值，也就是图例的名称
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    # loc代表了图例在整个坐标轴平面中的位置（一般选取'best'这个参数值）
    plt.legend(loc='best')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的loss值对比图")
    plt.savefig("images/{}".format(fname))
    plt.show()


# 准确率
def plot_acc(train_acc, val_acc,fname):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.xlabel('acc')
    plt.ylabel('epoch')
    plt.title("训练集和验证集的acc值对比图")
    plt.savefig("images/{}".format(fname))
    plt.show()

def matrix_and_kappa(y_pre,y_true,fname):
    y_pre = y_pre.cpu().numpy()
    y_pred = np.array(y_pre)
    y_pred_1 = []
    y_true_1 = []
    for i in y_pred:
        if i > 0.5:
            y_pred_1.append(1)
        else:
            y_pred_1.append(0)
    for i in y_true:
        if i == 1:
            y_true_1.append(1)
        else:
            y_true_1.append(0)
    C2 = confusion_matrix(y_true_1,y_pred_1)
    kappa_value = cohen_kappa_score(y_true_1, y_pred_1)
    f1 = f1_score(y_true_1,y_pred_1, average = 'macro')
    fpr,tpr,threshold1 = roc_curve(y_true_1,y_pred_1) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ##auc值

    # plt.figure() #绘制roc曲线
    # lw = 2
    # plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.savefig("images/{}".format(fname))
    # plt.show()
    return C2,kappa_value,f1,roc_auc

# 灵敏度(召回率)--sensitivity(true positive rate:TPR))-------specificity(true negative rate:TNR)--特异性
def plot_matrix(C2, labels_name, title=None, axis_labels=None,fname=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    cm = C2  # 生成混淆矩阵
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    sens = TP / float(TP+FN)
    spec = TN / float(TN+FP)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

#     plt.figure(figsize=(7, 5))
# # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
#     plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
#     plt.colorbar()  # 绘制图例
# # 图像标题
#     if title is not None:
#         plt.title(title,fontsize=15)
# # 绘制坐标
#     num_local = np.array(range(len(labels_name)))
#     if axis_labels is None:
#         axis_labels = labels_name
#     plt.xticks(num_local, axis_labels,fontsize=15, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
#     plt.yticks(num_local, axis_labels,fontsize=15)  # 将标签印在y轴坐标上
#     plt.ylabel('True label',fontsize=15)
#     plt.xlabel('Predicted label',fontsize=15)
#
#     thresh = cm.max() / 2.
#
#     # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
#     for i in range(np.shape(cm)[0]):
#        for j in range(np.shape(cm)[1]):
#            if int(cm[i][j] * 100 + 0.5) > 0:
#                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
#                        ha='center', va='center',
#                        color="white" if cm[i][j] > thresh else "black",fontsize=15)  # 如果要更改颜色风格，需要同时更改此行
#     # 显示
#     plt.savefig("images/{}".format(fname))
#     plt.show()
    return sens, spec
# plot_matrix(original_test_label[:,[1]],predictions1, [0, 1], title='confusion_matrix',
#            axis_labels=['left', 'right'])
def Tsne_visualize(features,labels,fname):
    X_tsne = TSNE(
        learning_rate=100,
        verbose=1,
        random_state=1234,
        init='pca',
        n_iter=1000
    ).fit_transform(features)
    y = np.squeeze(labels)
    y2 = np.argmax(y, axis=1)
    nclass = 2
    for i in range(nclass):
        ind = (y2 == i)
        plt.plot(X_tsne[ind, 0], X_tsne[ind, 1], '.', label=i + 1)
        plt.legend()
        plt.xticks(fontproperties='Times New Roman', size=16)
        plt.yticks(fontproperties='Times New Roman', size=16)
        plt.title('Visualization by tSNE', fontproperties='Times New Roman', size=16)
    plt.show()



# def visualize(features, labels, epoch):
#     colors = ["#ff0000", "#ffff00"]
#     plt.clf()
#     for i in range(2):
#         plt.plot(features[labels == i, 0], features[labels == i, 1], ".", c=colors[i])
#     plt.legend(["L", "R"], loc="upper right")
#     plt.title("epoch=%d" % epoch)
#     plt.savefig("images/epoch%d=" % epoch)
#     plt.show()


def nextBatch(data, label,batch_size):
    """
    Divide data into mini-batch
    """
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size) # 向上取值
    for idx in range(num_batches):
        start_idx = batch_size * idx
        end_idx = min(start_idx + batch_size, data_length)
        yield data[start_idx:end_idx],label[start_idx:end_idx]

def nextBatch_f(data1, data2,label,batch_size):
    """
    Divide data into mini-batch
    """
    data_length = len(data1)
    num_batches = math.ceil(data_length / batch_size) # 向上取值
    for idx in range(num_batches):
        start_idx = batch_size * idx
        end_idx = min(start_idx + batch_size, data_length)
        yield data1[start_idx:end_idx],data2[start_idx:end_idx],label[start_idx:end_idx]


# 定义 Grad-CAM 实现

if __name__ == "__main__":
    pass