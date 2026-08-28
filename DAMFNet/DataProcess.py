import torch
import numpy as np

# data_hbr = np.load('data\Raw_data\HbR\X_hbr12.npy')
# print(data_hbr.shape)
# data1 = data_hbr[:, :, 12:36]
# print(data_hbr.shape)
# data2 = data_hbr[:, :, 12:36]
def loading(i,data_path):  #下载数据集1
    print('load Raw data'+str(i))

    #EEG
    data_path1 = data_path + 'EEG/X_eeg'+str(i)+'.npy'
    label_path1 = data_path + 'EEG/Y_eeg'+str(i)+'.npy'
    data_eeg = np.load(data_path1)
    label_eeg = np.load(label_path1)

    data_path3 = data_path + 'HbO/X_hbo'+str(i)+'.npy'
    label_path3 = data_path + 'HbO/Y_hbo'+str(i)+'.npy'
    data_hbo = np.load(data_path3)
    data_hbo = data_hbo[:, 50:150, :]
    label_hbo = np.load(label_path3)
    #HbR
    data_path2 = data_path + 'HbR/X_hbr'+str(i)+'.npy'
    label_path2 = data_path + 'HbR/Y_hbr'+str(i)+'.npy'
    if i < 11:
        data_hbr = np.load(data_path2)
        data_hbr = data_hbr[:,50:150,:]
        label_hbr = np.load(label_path2)
    else:
        data_hbr = np.load(data_path2)
        data_hbr = data_hbr[:, :, 12:36]
        label_hbr= np.load(label_path2)



    return data_eeg, label_eeg, data_hbo, label_hbo, data_hbr, label_hbr

def loading2(i,data_path): #下载自己的数据集2
    print('load Raw data:', data_path)
    #EEG
    data_path1 = data_path + 'EEG/X_eeg'+str(i)+'.npy'
    label_path1 = data_path + 'EEG/Y_eeg'+str(i)+'.npy'
    data_eeg = np.load(data_path1)    #(120,2000,45)
    label_eeg = np.load(label_path1)   #(120,1)
    #HbR 、HbO
    data_path2 = data_path + 'fNIRS/X_hbr'+str(i)+'.npy'
    data_path3 = data_path + 'fNIRS/X_hbo' + str(i) + '.npy'
    label_path2 = data_path + 'fNIRS/Y_nirs'+str(i)+'.npy'
    data_hbr = np.load(data_path2)    #(120,100,51)
    data_hbo = np.load(data_path3)  # (120,100,51)
    label_hbr = np.load(label_path2)   #(120,1)

    return data_eeg, label_eeg, data_hbr, data_hbo,label_hbr


def sliding_window(data, label, fs, windows_long=3):
    X_data, y_data = [], []
    for i in range(data.shape[0]):
        data_t = data[i].T
        label_temp = label[i]
        X, y = [], []
        for start in range(8):  # 8 is the number of windows
            in_ = start * fs
            end = in_ + windows_long * fs
            train_seq = data_t[:, in_:end]
            train_seq = train_seq.T
            X.append(train_seq)
            # Convert one-hot to integer labels (0 or 1)
            y.append(np.argmax(label_temp))  # One-hot to integer conversion

        X_data.append(X)
        y_data.append(y)

    # Convert to numpy arrays
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # Check the shapes to avoid reshaping errors
    assert X_data.shape[0] == y_data.shape[0], "Mismatch between X and y shapes"

    # Reshaping data for the network
    X_data = X_data.reshape(-1, windows_long * fs, data.shape[-1])
    y_data = y_data.reshape(-1)  # Flatten labels to 1D (e.g., 0 or 1)

    X_data = torch.from_numpy(X_data)
    y_data = torch.from_numpy(y_data)

    return X_data, y_data



# 滑窗后一个被试-EEG：（60*8：600：8）（60*8：2）
# HbR（60*8：30：24） （60*8：2）
if __name__ == "__main__":
    pass
    # HbR
    # for s in range(1, 30):
    #     print(s)
    #     data_path='data/Raw_data/'
    #     data_path2 = data_path + 'HbR/X_hbr' + str(s) + '.npy'
    #     if s < 11:
    #         data_hbr = np.load(data_path2)
    #         data_hbr = data_hbr[:, 50:150, :]
    #     else:
    #         data_hbr = np.load(data_path2)
    #         data_hbr = data_hbr[:, :, 12:36]
    #
    #     print(data_hbr.shape)
    #     new_path = data_path + 'new_hbr/X_hbr' + str(s) + '.npy'
    #     np.save(new_path,data_hbr)

'''
    for i in range(1, 2):
        print(i)
        train_path = 'data/Raw_data/'
        data_eeg, label_eeg, data_hbr, label_hbr = loading(i, train_path)
        print(data_eeg.shape,label_eeg.shape)
'''


