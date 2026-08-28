# import os
# import torch
# import warnings
# # from configuration import args
# from func import nextBatch_f
# import random
# # ignore warning
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# warnings.filterwarnings('ignore')
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = 'cpu'
# #    fusion training
#
# # 定义函数，随机从0-end的一个序列中抽取size个不同的数
# def random_num(size, end):
#     range_ls = [i for i in range(end)]
#     num_ls = []
#     for i in range(size):
#         num = random.choice(range_ls)
#         range_ls.remove(num)
#         num_ls.append(num)
#     return num_ls
#
# def fusion_train(s_net,train_data1,train_data2, train_label, loss_fn, s_optimizer,args):
#
#     # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#
#     # print(f"Cuda:{device}")
#     s_net.train()
#     loss, current, n = 0.0, 0.0, 0
#     feature_loader = []
#     label_loader = []
#     for batch in nextBatch_f(train_data1, train_data2, train_label, args.batch_size):
#         x1 = batch[0]
#         x2 = batch[1]
#         y = batch[2]#-4
#         # print(x1.shape, y.shape)
#         # if args.iscuda:
#         # x1, x2, y = x1.to(device), x2.to(device), y.to(device)
#         (y1, y2, y_hat,ys_x1,ys_x2,s_1,s_2,pre_ts,ts,yt_x1,yt_x2,t_1,
#          t_2,pre_tt,tt,modality1_eeg,modality1_hbr)= s_net(x1,x2,modality1 = False, modality2 = False)
#         # print(y_hat.shape,  features.shape, y.shape)
#         y = y.long() #float()
#         s_loss = loss_fn(y_hat, y)
#         s1_loss = loss_fn(y1, y)
#         s2_loss = loss_fn(y2, y)
#         l = s1_loss + s2_loss + s_loss
#         _, pred = torch.max(y_hat, axis=1)
#         cur_acc = torch.sum(y == pred) / y_hat.shape[0]
#         # 反向传播
#         # 清空过往梯度
#         s_optimizer.zero_grad()
#         l.backward()
#         s_optimizer.step()
#         # feature_loader.append(features)
#         label_loader.append(y)
#         loss += l.item()
#         current += cur_acc.item()
#         n = n + 1
#
#     train_loss = loss / n
#     train_acc = current / n
#
#     # feature = torch.cat(feature_loader, dim=0)
#     labels = torch.cat(label_loader, dim=0)
#     # 计算训练的错误率,准确率
#     print('train_loss==' + str(train_loss), 'train_acc' + str(train_acc))
#
#     # return train_loss, train_acc, feature, labels
#     return train_loss, train_acc, labels
#
# def fusion_val(s_net,vali_data1, vali_data2, vali_label,loss_fn,args):
#     # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     loss, current, n = 0.0, 0.0, 0
#     # eval()：如果模型中有Batch Normalization和Dropout，则不启用，以防改变权值
#     s_net.eval()
#     with torch.no_grad():
#         for batch in nextBatch_f(vali_data1, vali_data2, vali_label, args.batch_size):
#             x1 = batch[0]
#             x2 = batch[1]
#             y = batch[2]#-4
#             # if args.iscuda:
#             # x1, x2, y = x1.to(device), x2.to(device), y.to(device)
#             # y1, y2, output, features1, features2, features, \
#             y1, y2, output,ys_x1,ys_x2,s_1,s_2,pre_ts,ts,yt_x1,\
#                 yt_x2,t_1,t_2,pre_tt,tt,modality1_eeg,modality1_hbr= s_net(x1,x2,modality1 = False, modality2 = False)
#             y = y.long() #float()
#             s_loss = loss_fn(output, y)
#             cur_loss = s_loss
#             _, pred = torch.max(output, axis=1)
#             cur_acc = torch.sum(y == pred) / output.shape[0]
#             loss += cur_loss.item()
#             current += cur_acc.item()
#             n = n + 1
#
#
#
#     val_loss = loss / n
#     val_acc = current / n
#     # 计算验证的错误率,准确率
#     print('val_loss=' + str(val_loss), 'val_acc=' + str(val_acc))
#
#     return val_loss, val_acc
#
#
# import os
# import torch
# import warnings
# import random
# from func import nextBatch_f
#
# # ignore warning
# warnings.filterwarnings('ignore')
#
#
# # Define function to randomly select `size` different numbers from 0 to `end`
# def random_num(size, end):
#     range_ls = [i for i in range(end)]
#     num_ls = []
#     for i in range(size):
#         num = random.choice(range_ls)
#         range_ls.remove(num)
#         num_ls.append(num)
#     return num_ls
#
#
# def fusion_train(s_net, train_data1, train_data2, train_label, loss_fn, s_optimizer, args):
#     s_net.train()
#     loss, current, n = 0.0, 0.0, 0
#     feature_loader = []
#     label_loader = []
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     for batch in nextBatch_f(train_data1, train_data2, train_label, args.batch_size):
#         # Move data to the same device (GPU or CPU)
#         x1 = batch[0].to(device)
#         x2 = batch[1].to(device)
#         y = batch[2].to(device)
#
#         # Forward pass
#         y1, y2, y_hat, ys_x1, ys_x2, s_1, s_2, pre_ts, ts, yt_x1, yt_x2, t_1, t_2, pre_tt, tt = s_net(
#             x1, x2)
#
#         # Ensure label is in long format (int)
#         y = y.long()
#
#         # Loss calculation
#         s_loss = loss_fn(y_hat, y)
#         s1_loss = loss_fn(y1, y)
#         s2_loss = loss_fn(y2, y)
#
#         # Total loss
#         total_loss = s1_loss + s2_loss + s_loss
#
#         # Accuracy calculation
#         _, pred = torch.max(y_hat, axis=1)
#         cur_acc = torch.sum(y == pred) / y_hat.shape[0]
#
#         # Backward pass and optimizer step
#         s_optimizer.zero_grad()
#         total_loss.backward()
#         s_optimizer.step()
#
#         label_loader.append(y)
#         loss += total_loss.item()
#         current += cur_acc.item()
#         n += 1
#
#     train_loss = loss / n
#     train_acc = current / n
#
#     labels = torch.cat(label_loader, dim=0)
#
#     # Print the training loss and accuracy
#     print('train_loss==' + str(train_loss), 'train_acc' + str(train_acc))
#
#     return train_loss, train_acc, labels
#
#
# def fusion_val(s_net, vali_data1, vali_data2, vali_label, loss_fn, args):
#     loss, current, n = 0.0, 0.0, 0
#     s_net.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     with torch.no_grad():
#         for batch in nextBatch_f(vali_data1, vali_data2, vali_label, args.batch_size):
#             x1 = batch[0].to(device)
#             x2 = batch[1].to(device)
#             y = batch[2].to(device)
#
#             # Forward pass
#             y1, y2, output, ys_x1, ys_x2, s_1, s_2, pre_ts, ts, yt_x1, yt_x2, t_1, t_2, pre_tt, tt = s_net(
#                 x1, x2)
#
#             # Ensure label is in long format (int)
#             y = y.long()
#
#             # Loss calculation
#             s_loss = loss_fn(output, y)
#             cur_loss = s_loss
#
#             # Accuracy calculation
#             _, pred = torch.max(output, axis=1)
#             cur_acc = torch.sum(y == pred) / output.shape[0]
#
#             loss += cur_loss.item()
#             current += cur_acc.item()
#             n += 1
#
#     val_loss = loss / n
#     val_acc = current / n
#
#     # Print the validation loss and accuracy
#     print('val_loss=' + str(val_loss), 'val_acc=' + str(val_acc))
#
#     return val_loss, val_acc

import os
import torch
import warnings
import random
from func import nextBatch_f

# ignore warning
warnings.filterwarnings('ignore')


# Define function to randomly select `size` different numbers from 0 to `end`
def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


def fusion_train(s_net, train_data1, train_data2, train_label, loss_fn, s_optimizer, args):
    s_net.train()
    loss, current, n = 0.0, 0.0, 0
    feature_loader = []
    label_loader = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch in nextBatch_f(train_data1, train_data2, train_label, args.batch_size):
        # Move data to the same device (GPU or CPU)
        x1 = batch[0].to(device)
        x2 = batch[1].to(device)
        y = batch[2].to(device)

        # Forward pass
        y1, y2, y_hat, ys_x1, ys_x2, s_1, s_2, pre_ts, ts, yt_x1, yt_x2, t_1, t_2, pre_tt, tt = s_net(
            x1, x2)

        # Ensure label is in long format (int)
        y = y.long()

        # Loss calculation
        s_loss = loss_fn(y_hat, y)
        s1_loss = loss_fn(y1, y)
        s2_loss = loss_fn(y2, y)

        w_eeg = float(getattr(args, "loss_w_eeg", 1.0))
        w_hbr = float(getattr(args, "loss_w_hbr", 1.0))
        w_fuse = float(getattr(args, "loss_w_fuse", 1.0))
        # Total loss
        total_loss = w_eeg * s1_loss + w_hbr * s2_loss + w_fuse * s_loss

        # Accuracy calculation
        _, pred = torch.max(y_hat, axis=1)
        cur_acc = torch.sum(y == pred) / y_hat.shape[0]

        # Backward pass and optimizer step
        s_optimizer.zero_grad()
        total_loss.backward()
        s_optimizer.step()

        label_loader.append(y)
        loss += total_loss.item()
        current += cur_acc.item()
        n += 1

    train_loss = loss / n
    train_acc = current / n

    labels = torch.cat(label_loader, dim=0)

    # Print the training loss and accuracy
    print('train_loss==' + str(train_loss), 'train_acc' + str(train_acc))

    return train_loss, train_acc, labels


def fusion_val(s_net, vali_data1, vali_data2, vali_label, loss_fn, args):
    loss, current, n = 0.0, 0.0, 0
    s_net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in nextBatch_f(vali_data1, vali_data2, vali_label, args.batch_size):
            x1 = batch[0].to(device)
            x2 = batch[1].to(device)
            y = batch[2].to(device)

            # Forward pass
            y1, y2, output, ys_x1, ys_x2, s_1, s_2, pre_ts, ts, yt_x1, yt_x2, t_1, t_2, pre_tt, tt = s_net(
                x1, x2)

            # Ensure label is in long format (int)
            y = y.long()

            # Loss calculation
            w_eeg = float(getattr(args, "loss_w_eeg", 1.0))
            w_hbr = float(getattr(args, "loss_w_hbr", 1.0))
            w_fuse = float(getattr(args, "loss_w_fuse", 1.0))

            s1_loss = loss_fn(y1, y)
            s2_loss = loss_fn(y2, y)
            s_loss  = loss_fn(output, y)

            cur_loss = w_eeg * s1_loss + w_hbr * s2_loss + w_fuse * s_loss

            # Accuracy calculation
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n += 1

    val_loss = loss / n
    val_acc = current / n

    # Print the validation loss and accuracy
    print('val_loss=' + str(val_loss), 'val_acc=' + str(val_acc))

    return val_loss, val_acc