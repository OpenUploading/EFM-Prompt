import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
import warnings
from configuration import args
from func import nextBatch

# ignore warning
warnings.filterwarnings('ignore')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = 'cpu'

def train(s_net,train_data, train_label, loss_fn, s_optimizer,args):
    s_net.train()
    loss, current, n = 0.0, 0.0, 0
    feature_loader = []
    label_loader = []
    for batch in nextBatch(train_data, train_label, args.batch_size):
        x = batch[0]
        y = batch[1]#-4

        # print(x.shape, y.shape)
        # if args.iscuda:
        #     x, y = x.to(device), y.to(device)
        y_hat, features= s_net(x)
        y = y.long()#.float()
        s_loss = loss_fn(y_hat, y)
        l= s_loss
        _, pred = torch.max(y_hat, axis=1)
        # print(y, pred)
        cur_acc = torch.sum(y == pred) / y_hat.shape[0] #y[:,1]
        # 反向传播
        # 清空过往梯度
        s_optimizer.zero_grad()
        l.backward()
        s_optimizer.step()
        feature_loader.append(features)
        label_loader.append(y)
        loss += l.item()
        current += cur_acc.item()
        n = n + 1
    train_loss = loss / n
    train_acc = current / n

    feature = torch.cat(feature_loader, dim=0)
    labels = torch.cat(label_loader, dim=0)
    # 计算训练的错误率,准确率
    print('train_loss==' + str(train_loss), 'train_acc = ' + str(train_acc))

    return train_loss, train_acc, feature, labels

def val(s_net,vali_eeg, vali_label,loss_fn,args):
    loss, current, n = 0.0, 0.0, 0
    # eval()：如果模型中有Batch Normalization和Dropout，则不启用，以防改变权值
    s_net.eval()
    with torch.no_grad():
        for batch in nextBatch(vali_eeg, vali_label, args.batch_size):
            x = batch[0]
            y = batch[1]#-4
            # if args.iscuda:
            #     x, y = x.to(device), y.to(device)
            output, features = s_net(x)
            y = y.long() #.float()
            s_loss = loss_fn(output, y)
            cur_loss = s_loss
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    # 计算验证的错误率,准确率
    print('val_loss=' + str(val_loss), 'val_acc=' + str(val_acc))
    return val_loss, val_acc