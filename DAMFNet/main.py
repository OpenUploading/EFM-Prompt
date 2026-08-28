import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
import warnings
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model.eeg_net import MYEEG
from model.Hbr_net import MYHBR
from model.fusion_net import MYFusion
from DataProcess import loading,sliding_window,loading2, loading3
# ,loading2)
from sklearn.model_selection import train_test_split
# from configuration import args
from func4 import  plot_loss, plot_acc, matrix_and_kappa
from training import train,val
from fusion_training import fusion_train,fusion_val,random_num
import csv
import time
from matplotlib import pyplot as plt
import argparse

#ignore warning
warnings.filterwarnings('ignore')
#
# device = torch.device("cuda " if torch.cuda.is_available() else "cpu")
# device = 'cpu'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="fusion_net",
                        choices=["eeg_net", "Hbr_net", "fusion_net", "EEGNet", "M2NN"])
    parser.add_argument('--gpu', type=str, default="0", help="GPU ID(s) to use, e.g., '0' or '0,1,2'")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--drop_prob", type=float, default=0.4)  # 0.2~0.5调节
    parser.add_argument("--iscuda", type=bool, default=False)
    parser.add_argument("--hidden_size", type=int, default=1024)
    # hbr
    parser.add_argument("--hidden_size_hbr", type=int, default=1024)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    for s in range(1,2):
        # s = s+1   #subject 1~29  数据2 1~16
        print(s)
        # s = 1
        #:data_eeg, label_eeg, data_hbr, label_hbr
        eeg, label, hbo, hbo_label, hbr, hbr_label = loading(s, data_path='data/Raw_data/')  #下载数据集1
        # eeg, label, hbr, hbo, hbr_label = loading2(s, data_path='data/Mydata/') # 下载数据集2 data\Mydata
        # eeg = eeg[:, :, 0:30]
        # hbr = hbr[:, :, 0:34]
        # label = np.squeeze(label)
        # hbr_label = np.squeeze(hbr_label)
        # print(eeg.shape, label.shape, hbr.shape, hbo.shape, hbr_label.shape)
        # k-fold cross validation
        k = 10
        num_test_samples = len(eeg) // k
        seed = 2
        for fold in range(k):
            # 划分数据集1
            test_eeg = eeg[num_test_samples * fold:num_test_samples * (fold + 1)]
            test_label1 = label[num_test_samples * fold:num_test_samples * (fold + 1)]
            new_eeg = np.concatenate((eeg[:num_test_samples * fold], eeg[num_test_samples * (fold + 1):]), axis=0)
            new_label = np.concatenate((label[:num_test_samples * fold], label[num_test_samples * (fold + 1):]), axis=0)

            test_hbr = hbr[num_test_samples * fold:num_test_samples * (fold + 1)]
            test_label2 = hbr_label[num_test_samples * fold:num_test_samples * (fold + 1)]
            new_hbr = np.concatenate((hbr[:num_test_samples * fold], hbr[num_test_samples * (fold + 1):]), axis=0)
            new_label2 = np.concatenate((hbr_label[:num_test_samples * fold], hbr_label[num_test_samples * (fold + 1):]),
                                        axis=0)
            train_eeg, val_eeg, train_label1, val_label1 = train_test_split(new_eeg, new_label, test_size=0.1,
                                                                                         shuffle=True,
                                                                                         stratify=new_label,
                                                                                         random_state=seed)  #
            train_hbr, val_hbr, train_label2, val_label2 = train_test_split(new_hbr, new_label2,
                                                                                          test_size=0.1,
                                                                                          shuffle=True,
                                                                                          stratify=new_label2,
                                                                                          random_state=seed)  #
            # print('eeg_label', test_label, 'hbr_label', test_label2)
            # print('eeg_label', validation_label, 'hbr_label', validation_label2)

            #  划分数据集2
            # testing
            # new_eeg, test_eeg, new_label1, test_label1 = train_test_split(eeg, label,
            #                                                              test_size=0.1,
            #                                                              shuffle=True,
            #                                                              stratify=label,
            #                                                              random_state=fold
            #                                                               )
            # # print(test_label1.shape,test_label1)
            # new_hbr, test_hbr, new_label2, test_label2 = train_test_split(hbr, hbr_label,
            #                                                              test_size=0.1,
            #                                                              shuffle=True,
            #                                                              stratify=hbr_label,
            #                                                              random_state=fold)
            # # print(test_label2)
            # new_hbo, test_hbo, new_label3, test_label3 = train_test_split(hbo, hbr_label,
            #                                                              test_size=0.1,
            #                                                              shuffle=True,
            #                                                              stratify=hbr_label,
            #                                                              random_state=fold)
            # # print('new_eeg/', new_eeg.shape, 'test_eeg/', test_eeg.shape)
            # # print('test_eeg_label/', test_label1, 'test_hbr_label/', test_label2,'test_hbo_label/', test_label3)
            #
            # # validation
            # train_eeg, val_eeg, train_label1, val_label1 = train_test_split(new_eeg, new_label1,
            #                                                                  test_size=0.1,
            #                                                                  shuffle=True,
            #                                                                  stratify=new_label1,
            #                                                                  random_state=fold
            #                                                                 )
            # # print(val_label1.shape,val_label1)
            # train_hbr, val_hbr, train_label2, val_label2 = train_test_split(new_hbr, new_label2,
            #                                                              test_size=0.1,
            #                                                              shuffle=True,
            #                                                              stratify=new_label2,
            #                                                              random_state=fold)
            # train_hbo, val_hbo, train_label3, val_label3 = train_test_split(new_hbo, new_label3,
            #                                                              test_size=0.1,
            #                                                              shuffle=True,
            #                                                              stratify=new_label3,
            #                                                              random_state=fold)

            # sliding_3s
            train_eeg, train_label1 = sliding_window(train_eeg, train_label1, 200)
            val_eeg, val_label1 = sliding_window(val_eeg, val_label1, 200)
            test_eeg, test_label1 = sliding_window(test_eeg, test_label1, 200)

            train_hbr, train_label2 = sliding_window(train_hbr, train_label2, 10)
            val_hbr, val_label2 = sliding_window(val_hbr, val_label2, 10)
            test_hbr, test_label2 = sliding_window(test_hbr, test_label2, 10)

            # train_hbo, train_label3 = sliding_window(train_hbo, train_label3, 10)
            # val_hbo, val_label3 = sliding_window(val_hbo, val_label3, 10)
            # test_hbo, test_label3 = sliding_window(test_hbo, test_label3, 10)

            # print('test_eeg_label/', test_label1, 'test_hbr_label/', test_label2)

            # shuffle
            index_validation = [i for i in range(len(val_eeg))]
            np.random.shuffle(index_validation)
            vali_eeg = val_eeg[index_validation]
            vali_hbr = val_hbr[index_validation]
            # vali_hbo = val_hbo[index_validation]
            vali_label = val_label1[index_validation]
            vali_label2 = val_label2[index_validation]
            # vali_label3 = val_label3[index_validation]

            index_train = [i for i in range(len(train_eeg))]
            np.random.shuffle(index_train)
            train_eeg = train_eeg[index_train]
            train_hbr = train_hbr[index_train]
            # train_hbo = train_hbo[index_train]
            train_label = train_label1[index_train]
            train_label2 = train_label2[index_train]
            # train_label3 = train_label3[index_train]

            index_test = [i for i in range(len(test_eeg))]
            np.random.shuffle(index_test)
            test_eeg = test_eeg[index_test]
            test_hbr = test_hbr[index_test]
            # test_hbo = test_hbo[index_test]
            test_label = test_label1[index_test]
            test_label2 = test_label2[index_test]
            # test_label3 = test_label3[index_test]
            print("loading dataset done")

            train_eeg, train_label = train_eeg.to(device), train_label.to(device)
            train_hbr = train_hbr.to(device)
            # train_hbo = train_hbo.to(device)

            # print('test_eeg_label/', test_label, 'test_hbr_label/', test_label2)
            # print('vali_eeg_label/', vali_label, 'vali_hbr_label/', vali_label2,'vali_hbo_label/', vali_label3)
            # 模型
            if args.model == "eeg_net":
                s_net = MYEEG(
                    in_channel=1,
                    out_channels=[16, 32, 32, 64],
                    #   input_size=4800,
                    hidden_size=args.hidden_size,
                    output_size=1024,
                    drop_prob=args.drop_prob).to(device)
            elif args.model == "Hbr_net":
                s_net = MYHBR(
                    in_channel=1,
                    out_channels=[16, 16, 32],
                    #   input_size=4800,
                    hidden_size=args.hidden_size_hbr,
                    drop_prob=args.drop_prob).to(device)
                # s_net = HbrNet(in_channels = 1).to(device)
                # s_net = EEGNet().to(device)


            elif args.model == "fusion_net":
                s_net = MYFusion().to(device)

            # summary(s_net,[(1,600,8),(1,30,24)])
            # flops, params = profile(s_net, inputs=(train_eeg, train_hbr))
            # print(f"估算的 FLOPs 数量: {flops / 1e6} 百万")
            # print(f"模型参数数量: {params / 1e6} 百万")

            print("loading model {} done".format(args.model))

            s_optimizer = optim.Adam(params=s_net.parameters(),lr=args.lr) #,betas=(0.9, 0.999)
            loss_fn = nn.CrossEntropyLoss()  #数据集1 nn.BCELoss()/nn.BCEWithLogitsLoss  数据集2 nn.CrossEntropyLoss()

            loss_train = []
            acc_train = []
            loss_val = []
            acc_val = []
            # 用来记录最优的正确率
            best_acc = 0.0
            iteration = 0
            train_times = 0.0
            t0 = time.time()
            for i in range(args.epochs):
                t1 = time.time()
                print("=========epoch {}=========".format(i + 1))
                epoch = i
                # training
                if args.model == "eeg_net":
                    train_loss, train_acc,feature,labels = train(s_net,train_eeg, train_label,
                                                                 loss_fn, s_optimizer,args)
                    # 验证模型
                    vali_eeg, vali_label = vali_eeg.to(device), vali_label.to(device)
                    val_loss, val_acc = val(s_net,vali_eeg, vali_label, loss_fn,args)

                    loss_train.append(train_loss)
                    acc_train.append(train_acc)
                    loss_val.append(val_loss)
                    acc_val.append(val_acc)

                    # 保存最好的模型权重
                    if val_acc > best_acc:
                        folder = 'save_model/eeg'  # 设置模型存储路径
                        # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                        if not os.path.exists(folder):
                            # os.mkdir() 方法用于以数字权限模式创建目录
                            os.mkdir('save_model/eeg')
                        best_acc = val_acc
                        print(f"save best model，第{i + 1}轮", i + 1)
                        # torch.save(state, dir)：保存模型等相关参数，dir表示保存文件的路径+保存文件名
                        # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
                        torch.save(s_net.state_dict(), 'save_model/eeg/s_net'+str(s) + '_' + str(fold)+'.pth')

                elif args.model == "Hbr_net":
                    train_nirs = train_hbr
                    vali_nirs = vali_hbr
                    print("Traing set of nirs:")
                    train_loss, train_acc,feature,labels = train(s_net,train_nirs,train_label,
                                                                 loss_fn, s_optimizer,args)
                    # 验证模型
                    vali_nirs, vali_label = vali_nirs.to(device), vali_label.to(device)
                    val_loss, val_acc = val(s_net,vali_nirs, vali_label, loss_fn,args)

                    loss_train.append(train_loss)
                    acc_train.append(train_acc)
                    loss_val.append(val_loss)
                    acc_val.append(val_acc)

                    # 保存最好的模型权重
                    if val_acc > best_acc:
                        folder = 'save_model/hbr'# 设置模型存储路径
                        # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                        if not os.path.exists(folder):
                            # os.mkdir() 方法用于以数字权限模式创建目录
                            os.mkdir('save_model/hbr')
                        best_acc = val_acc
                        print(f"save best model，第{i + 1}轮",i + 1)
                        # torch.save(state, dir)：保存模型等相关参数，dir表示保存文件的路径+保存文件名
                        # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
                        torch.save(s_net.state_dict(), 'save_model/hbr/s_net'+str(s) + '_' + str(fold)+'.pth')

                elif args.model == "fusion_net":
                    train_loss, train_acc,labels = fusion_train(s_net,train_eeg,train_hbr, train_label,#feature,
                                                                 loss_fn, s_optimizer,args)
                    # 验证模型
                    vali_eeg, vali_label = vali_eeg.to(device), vali_label.to(device)
                    vali_hbr = vali_hbr.to(device)
                    val_loss, val_acc = fusion_val(s_net,vali_eeg,vali_hbr, vali_label, loss_fn,args)

                    loss_train.append(train_loss)
                    acc_train.append(train_acc)
                    loss_val.append(val_loss)
                    acc_val.append(val_acc)

                    # 保存最好的模型权重
                    if val_acc >best_acc:
                        folder = 'save_model/fusion2'# 设置模型存储路径
                        # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                        if not os.path.exists(folder):
                            # os.mkdir() 方法用于以数字权限模式创建目录
                            os.mkdir('save_model/fusion2')
                        best_acc = val_acc
                        print(f"save best model，第{i + 1}轮",i + 1)
                        # torch.save(state, dir)：保存模型等相关参数，dir表示保存文件的路径+保存文件名
                        # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
                        torch.save(s_net.state_dict(), 'save_model/fusion2/s_net'+str(s) + '_' + str(fold)+'.pth')

                t2 = time.time()
                training_time = t2 - t1
                print('training_time==' + str(training_time))
            t3 = time.time()
            all_training_time = t3 - t0
            print('all_training_time==' + str(all_training_time))

            if args.model == "eeg_net":
                folder = 'images/eeg'  # 设置模型存储路径
                # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                if not os.path.exists(folder):
                    # os.mkdir() 方法用于以数字权限模式创建目录
                    os.mkdir('images/eeg')
                fname = 'eeg/subject0' + str(s) + '_' + str(fold)  # 被试

                plot_loss(loss_train, loss_val,fname +'_loss1')
                plot_acc(acc_train, acc_val,fname +'_acc1')
                print('Training done!')
                print('Testing star...')
            elif args.model == "Hbr_net":
                folder = 'images/hbr'  # 设置模型存储路径
                # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                if not os.path.exists(folder):
                    # os.mkdir() 方法用于以数字权限模式创建目录
                    os.mkdir('images/hbr')
                fname = 'hbr/subject0' + str(s) + '_' + str(fold)
                plot_loss(loss_train, loss_val, fname + '_loss2')
                plot_acc(acc_train, acc_val, fname + '_acc2')
                print('Training done!')
                print('Testing star...')

            elif args.model == "fusion_net":
                folder = 'images/fusion'  # 设置模型存储路径
                # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                if not os.path.exists(folder):
                    # os.mkdir() 方法用于以数字权限模式创建目录
                    os.mkdir('images/fusion')
                fname = 'fusion/subject0' + str(s) + '_' + str(fold)
                plot_loss(loss_train, loss_val, fname + '_loss3')
                plot_acc(acc_train, acc_val, fname + '_acc3')
                print('Training done!')
                print('Testing star...')


            x1 = test_eeg  ##特征可视化
            x2 = test_hbr #hbr
            y = test_label
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y = y-4
            t4 = time.time()
            # testing
            if args.model == "eeg_net":
                s_net = MYEEG(
                    in_channel=1,
                    out_channels=[16, 32, 32, 64],
                    hidden_size=args.hidden_size,
                    output_size=1024,
                    drop_prob=args.drop_prob).to(device)

                s_net.load_state_dict(torch.load(r'save_model/eeg/s_net'+str(s) + '_' + str(fold)+'.pth'))
                s_net.eval()

                with torch.no_grad():
                    # output, features = s_net(x1)
                    output,features = s_net(x1)
                    print(output.shape)
                    y = y.long()#.float()
                    s_loss = loss_fn(output, y)
                    test_loss = s_loss
                    _, pred = torch.max(output, axis=1)
                    test_acc = torch.sum(y == pred) / output.shape[0]

                # 计算验证的错误率,准确率,保存
                print('test_loss=' + str(test_loss), 'test_acc=' + str(test_acc))
                file = open('test_eeg_16.csv', 'a+', newline='', encoding='utf-8')
                swriter = csv.writer(file)
                swriter.writerow([str(fold+1),'test_loss', str(test_loss), 'test_acc', str(test_acc)])
                # matrix，kappa,f1-score,roc---
                folder = 'images/roc1'  # 设置模型存储路径
                # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                if not os.path.exists(folder):
                    # os.mkdir() 方法用于以数字权限模式创建目录
                    os.mkdir('images/roc1')
                fname = 'roc1/subject0' + str(s) + '_' + str(fold)  # 被试
                C2, kappa_value, f1, roc_auc = matrix_and_kappa(output, y, fname+'_roc1')
                folder = 'images/matrix1'  # 设置模型存储路径
                # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                if not os.path.exists(folder):
                    # os.mkdir() 方法用于以数字权限模式创建目录
                    os.mkdir('images/matrix1')
                path = 'matrix1/subject0' + str(s) + '_' + str(fold)  # 被试
                sens, spec = plot_matrix(C2,labels_name = [0, 1,2,3],title='confusion_matrix',
                                         axis_labels=['Left', 'down','Left-right','right-left'], fname=path + '_matrix1')
                files = open('evalution_eeg_16.csv', 'a+', newline='', encoding='utf-8')
                swriter = csv.writer(files)
                swriter.writerow([str(fold + 1), 'sensitivity', str(sens), 'specificity', str(spec),
                                  'kappa', str(kappa_value),'f1-score', str(f1)])
            elif args.model == "Hbr_net":
                s_net = MYHBR(
                    in_channel=1,
                    out_channels=[16, 16, 32],
                    #   input_size=4800,
                    hidden_size=args.hidden_size_hbr,
                    drop_prob=args.drop_prob).to(device)

                s_net.load_state_dict(torch.load(r'save_model/hbr/s_net'+str(s) + '_' + str(fold)+'.pth'))
                s_net.eval()

                with torch.no_grad():

                    output, features = s_net(x2)
                    y = y.long()#float()
                    s_loss = loss_fn(output, y)
                    test_loss = s_loss
                    _, pred = torch.max(output, axis=1)
                    test_acc = torch.sum(y == pred) / output.shape[0]
                # 计算验证的错误率,准确率
                print('test_loss=' + str(test_loss), 'test_acc=' + str(test_acc))

                file = open('test_hbr_16.csv', 'a+', newline='', encoding='utf-8')
                swriter = csv.writer(file)
                swriter.writerow([str(fold+1),'test_loss', str(test_loss), 'test_acc', str(test_acc)])

                # matrix，kappa,f1-score,roc---
                folder = 'images/roc2'  # 设置模型存储路径
                # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                if not os.path.exists(folder):
                    # os.mkdir() 方法用于以数字权限模式创建目录
                    os.mkdir('images/roc2')
                fname = 'roc2/subject0' + str(s) + '_' + str(fold)  # 被试
                C2, kappa_value, f1, roc_auc= matrix_and_kappa(output, y, fname+'_roc2') #, roc_auc
                folder = 'images/matrix2'  # 设置模型存储路径
                # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                if not os.path.exists(folder):
                    # os.mkdir() 方法用于以数字权限模式创建目录
                    os.mkdir('images/matrix2')
                path = 'matrix2/subject0' + str(s) + '_' + str(fold)  # 被试
                sens, spec = plot_matrix(C2,labels_name = [0, 1, 3, 4],title='confusion_matrix',
                                         axis_labels=['Left', 'down','Left-right','right-left'], fname=path + '_matrix2')
                files = open('evalution_hbr_16.csv', 'a+', newline='', encoding='utf-8')
                swriter = csv.writer(files)
                swriter.writerow([str(fold + 1), 'sensitivity', str(sens), 'specificity', str(spec),
                                  'kappa', str(kappa_value),'f1-score', str(f1)])
            elif args.model == "fusion_net":
                t4 = time.time()
                s_net = MYFusion().to(device)
                s_net.load_state_dict(torch.load(r'save_model/fusion2/s_net' + str(s) + '_' + str(fold) + '.pth'))
                s_net.eval()

                with torch.no_grad():
                    # out1, out2, out_fusion, features1,features2,features, \
                    out1, out2, out_fusion,ys_x1,ys_x2,s_1,s_2,pre_ts,ts,yt_x1\
                        ,yt_x2,t_1,t_2,pre_tt,tt,modality1_eeg,modality1_hbr = s_net(x1, x2,modality1 = False, modality2 = False)
                    y = y.long() #float()
                    s1_loss = loss_fn(out1, y)
                    s2_loss = loss_fn(out2, y)
                    s_loss = loss_fn(out_fusion, y)

                    test_loss = s1_loss + s2_loss + s_loss
                    _, pred = torch.max(out_fusion, axis=1)
                    test_acc = torch.sum(y == pred) / out_fusion.shape[0]
                    t5 = time.time()
                    testing_time = t5 - t4
                    print('testing_time==' + str(testing_time))

                    v1 = modality1_eeg[1]
                    v1 = v1.data.squeeze(0)
                    v1 = v1.mean(dim=0)#[1]
                    v1 = v1.data.squeeze(0)
                    # 假设 v 是一个存储在 GPU 上的张量
                    v1_cpu = v1.cpu()  # 将张量从 GPU 移动到 CPU
                    print(v1_cpu.shape)  # torch.Size([512, 28, 28])
                    # channel_num = random_num(4, v_cpu.shape[0])
                    plt.figure(figsize=(10, 10))
                    # for index, channel in enumerate(channel_num):
                    #     ax = plt.subplot(1, 4, index + 1, )
                    plt.imshow(v1_cpu[:, :].T)
                    plt.xticks()  # 设置x轴的刻度
                    plt.yticks([])  # 设置y轴的刻度
                    plt.colorbar(fraction=0.046, pad=0.04, shrink=0.5, aspect=5)
                    feature_name = os.path.join('Features', f'{s}_{fold+1}')
                    os.makedirs(os.path.dirname(feature_name), exist_ok=True)
                    plt.savefig(feature_name, dpi=300)

                    v2 = modality1_hbr[1]
                    v2 = v2.data.squeeze(0)
                    v2 = v2.mean(dim=0)#[1]
                    v2 = v2.data.squeeze(0)
                    # 假设 v 是一个存储在 GPU 上的张量
                    v2_cpu = v2.cpu()  # 将张量从 GPU 移动到 CPU
                    print(v2_cpu.shape)  # torch.Size([512, 28, 28])
                    # channel_num = random_num(4, v_cpu.shape[0])
                    plt.figure(figsize=(10, 10))
                    # for index, channel in enumerate(channel_num):
                    #     ax = plt.subplot(1, 4, index + 1, )
                    plt.imshow(v2_cpu[:, :].T)
                    plt.xticks()  # 设置x轴的刻度
                    plt.yticks([])  # 设置y轴的刻度
                    plt.colorbar(fraction=0.046, pad=0.04, shrink=0.5, aspect=5)
                    feature_name = os.path.join('Features2', f'{s}_{fold+1}')
                    os.makedirs(os.path.dirname(feature_name), exist_ok=True)
                    plt.savefig(feature_name, dpi=300)

                # 计算验证的错误率,准确率
                print('test_loss=' + str(test_loss), 'test_acc=' + str(test_acc))
                # ys_x1 = np.array(ys_x1)
                # ys_x2 = np.array(ys_x2)
                # s_1 = np.array(s_1)
                # s_2 = np.array(s_2)
                # yt_x1 = np.array(yt_x1)
                # yt_x2 = np.array(yt_x2)
                # t_1 = np.array(t_1)
                # t_1 = np.array(t_2)
                # torch.save(ys_x1, 'data/Att/ys_x1'+str(fold)+'.pt')
                # torch.save(ys_x2, 'data/Att/ys_x2'+str(fold)+'.pt')
                # torch.save(s_1, 'data/Att/s_1'+str(fold)+'.pt')
                # torch.save(s_2, 'data/Att/s_2'+str(fold)+'.pt')
                # torch.save(pre_ts, 'data/Att/pre_ts' + str(fold) + '.pt')
                # torch.save(ts, 'data/Att/ts' + str(fold) + '.pt')
                # torch.save(yt_x1, 'data/Att/yt_x1'+str(fold)+'.pt')
                # torch.save(yt_x2, 'data/Att/yt_x2'+str(fold)+'.pt')
                # torch.save(t_1, 'data/Att/t_1'+str(fold)+'.pt')
                # torch.save(t_2, 'data/Att/t_2'+str(fold)+'.pt')
                # torch.save(pre_tt, 'data/Att/pre_tt ' + str(fold) + '.pt')
                # torch.save(tt, 'data/Att/tt ' + str(fold) + '.pt')

                # file = open('test_fusion_16.csv', 'a+', newline='', encoding='utf-8')
                # swriter = csv.writer(file)
                # swriter.writerow([str(fold + 1), 'test_loss', str(test_loss), 'test_acc', str(test_acc)])
                #
                # # matrix，kappa,f1-score,roc---
                # folder = 'images/roc'  # 设置模型存储路径
                # # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                # if not os.path.exists(folder):
                #     # os.mkdir() 方法用于以数字权限模式创建目录
                #     os.mkdir('images/roc')
                # fname = 'roc/subject0' + str(s) + '_' + str(fold)  # 被试
                # C2, kappa_value, f1, roc_auc = matrix_and_kappa(out_fusion, y, fname+'_roc')
                #
                # folder = 'images/matrix'  # 设置模型存储路径
                # # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
                # if not os.path.exists(folder):
                #     # os.mkdir() 方法用于以数字权限模式创建目录
                #     os.mkdir('images/matrix')
                # path = 'matrix/subject0' + str(s) + '_' + str(fold)  # 被试
                # sens, spec = plot_matrix(C2,labels_name = [0, 1, 2, 3],title='confusion_matrix',
                #                          axis_labels=['Left', 'down','Left-right','right-left'], fname=path + '_matrix')
                # files = open('evalution_fusion_16.csv', 'a+', newline='', encoding='utf-8')
                # swriter = csv.writer(files)
                # swriter.writerow([str(fold + 1), 'sensitivity', str(sens), 'specificity', str(spec),
            #                       'kappa', str(kappa_value),'f1-score', str(f1)])
            # t5 = time.time()
            # testing_time = t5-t4
            # print('testing_time==' + str(testing_time))