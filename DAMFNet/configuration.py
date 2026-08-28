import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="fusion_net",
                    choices=["eeg_net","Hbr_net","fusion_net","EEGNet","M2NN"])
parser.add_argument("--batch_size", type=int, default=40)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--drop_prob", type=float, default=0.4)  # 0.2~0.5调节
parser.add_argument("--iscuda",type=bool,default=False)
parser.add_argument("--hidden_size",type=int,default=1024)
#hbr
parser.add_argument("--hidden_size_hbr",type=int,default=1024)
args = parser.parse_args()

if __name__ == "__main__":
    pass