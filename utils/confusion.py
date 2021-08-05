import torch
import numpy as np

class ConfusionMatrix():
    def __init__(self, num_classes):
        self.total = 0
        self.target_num = torch.zeros((1, num_classes))
        self.predict_num = torch.zeros((1, num_classes))
        self.acc_num = torch.zeros((1, num_classes))
        self.recall = torch.zeros((1, num_classes))
        self.precision = torch.zeros((1, num_classes))
        self.F1 = torch.zeros((1, num_classes))
        self.val_accuracy = 0

        self.mean_recall = 0
        self.mean_precision = 0
        self.mean_F1 = 0
        self.mean_val_accuracy = 0

    def update(self, val_labels, outputs, predict_y):
        self.total += val_labels.size(0)
        pre_mask = torch.zeros(outputs.size()).scatter_(1, predict_y.cpu().view(-1, 1), 1.)
        self.predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(1, val_labels.data.cpu().view(-1, 1), 1.)
        self.target_num += tar_mask.sum(0)
        acc_mask = pre_mask * tar_mask
        self.acc_num += acc_mask.sum(0)

    def acc_p_r_f1(self):
        self.recall = self.acc_num / self.target_num
        self.precision = self.acc_num / self.predict_num
        self.F1 = 2 * self.recall * self.precision / (self.recall + self.precision)
        self.accuracy = self.acc_num.sum(1) / self.target_num.sum(1)
        # 精度调整
        self.recall = np.nan_to_num((self.recall.numpy()[0] * 100))
        self.precision = np.nan_to_num((self.precision.numpy()[0] * 100))
        self.F1 = np.nan_to_num((self.F1.numpy()[0] * 100))
        self.val_accuracy = np.nan_to_num((self.accuracy.numpy()[0] * 100))

        self.mean_val_accuracy = self.val_accuracy.mean()
        self.mean_precision = self.precision.mean()
        self.mean_recall = self.recall.mean()
        self.mean_F1 = self.F1.mean()

        # return val_accuracy.mean(), precision.mean(), recall.mean(), F1.mean()

    def save(self, results_file, epoch):
        self.precision = ['%7.3f'%i for i in self.precision]
        self.recall = ['%7.3f'%i for i in self.recall]
        self.F1 = ['%7.3f'%i for i in self.F1]
        with open(results_file, 'a') as f:
            f.write('\n%5d'%epoch+' '+'%8.3f'%self.mean_val_accuracy+' '+'%9.3f'%self.mean_precision+' '+'%6.3f'%self.mean_recall+' '\
                    +'%8.3f'%self.mean_F1+' '+' '.join(self.precision)+' '+' '.join(self.recall)+' '+' '.join(self.F1))
        # 打印格式方便复制
        # print('recall', " ".join('%s' % id for id in recall))
        # print('precision', " ".join('%s' % id for id in precision))
        # print('F1', " ".join('%s' % id for id in F1))
