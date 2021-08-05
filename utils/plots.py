import os
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

def plot_base(ax, epochs, y, color, title=None):
    ax.plot(epochs, y, color=color)
    ax.grid(axis='y')
    ax.grid(axis='x')
    if title:
        ax.set_title(title)
    return ax

def double_bar(num_train_class, num_val_class, classes, log_dir):
    width = 0.75  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(classes, num_train_class, width, label='train')
    ax.bar(classes, num_val_class, width, bottom=num_train_class, label='val')
    ax.set_ylabel('Number')
    ax.set_title('Number by train/val and class')
    ax.legend(bbox_to_anchor=(-0.15, 0.7), fontsize=5)
    plt.savefig(os.path.join(log_dir, 'data_distribution.jpg'), dpi=600, bbox_inches='tight')
    # plt.show()

def plot_datasets(img_path, log_dir):
    train_root = os.path.join(img_path, 'train')
    val_root = os.path.join(img_path, 'val')
    classes = os.listdir(train_root)
    num_train_class = [len(os.listdir(os.path.join(train_root, class_))) for class_ in classes]
    num_val_class = [len(os.listdir(os.path.join(val_root, class_))) for class_ in classes]
    double_bar(num_train_class, num_val_class, classes, log_dir)

def plot_txt(log_dir, num_classes, labels_name):
    txt_results = np.loadtxt(os.path.join(log_dir, "results.txt"), dtype=str, delimiter=',')
    txt_results = [[i for i in j.split(' ')] for j in txt_results]
    for i in txt_results:
        while '' in i:
            i.remove('')

    txt_results = np.array(txt_results)
    epochs = txt_results[:, 0][1:].astype(int)
    accuracy = txt_results[:, 1][1:].astype(float)
    precision = txt_results[:, 2][1:].astype(float)
    recall = txt_results[:, 3][1:].astype(float)
    F1 = txt_results[:, 4][1:].astype(float)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax[0, 0] = plot_base(ax[0, 0], epochs, accuracy, color='red', title='accuracy')
    ax[0, 1] = plot_base(ax[0, 1], epochs, precision, color='red', title='precision')
    ax[1, 0] = plot_base(ax[1, 0], epochs, recall, color='red', title='recall')
    ax[1, 1] = plot_base(ax[1, 1], epochs, F1, color='red', title='F1')
    fig.text(0.5, 0.04, 'Epoch', ha='center')
    plt.savefig(os.path.join(log_dir, 'Acc-P-R-F1.jpg'), dpi=600, bbox_inches='tight')

    fig, ax = plt.subplots(3, num_classes, figsize=(2 * num_classes, 2 * 3), sharex=True, sharey=True)
    for i in range(num_classes):
        precision_i = txt_results[:, 5 + i][1:].astype(float)
        recall_i = txt_results[:, 5 + num_classes + i][1:].astype(float)
        F1_i = txt_results[:, 5 + 2 * num_classes + i][1:].astype(float)

        ax[0, i] = plot_base(ax[0, i], epochs, precision_i, color='blue')
        ax[0, i].set_title(labels_name[i])
        ax[1, i] = plot_base(ax[1, i], epochs, recall_i, color='blue')
        ax[2, i] = plot_base(ax[2, i], epochs, F1_i, color='blue')

    ax[0, 0].set_ylabel('Precision')
    ax[1, 0].set_ylabel('Recall')
    ax[2, 0].set_ylabel('F1-score')
    fig.text(0.5, 0.04, 'Epoch', ha='center')
    plt.savefig(os.path.join(log_dir, 'P-R-F1-per-class.jpg'), dpi=600, bbox_inches='tight')


def plot_lr_scheduler(optimizer, scheduler, epochs, log_dir, scheduler_type):
    optimizer, scheduler = copy(optimizer), copy(scheduler)
    plt.figure()
    y = []
    for epoch in range(epochs):
        y.append(optimizer.param_groups[0]['lr'])
        # print('epoch:', epoch, 'lr:', optimizer.param_groups[0]['lr'])
        scheduler.step()
    plt.plot(y, c='r', label='warmup step_lr', linewidth=1)
    plt.legend(loc='best')
    # plt.xticks(np.arange(0, epochs+20, 20))
    if scheduler_type!='warmup_cosine_lr':
        plt.yscale("log")
    plt.savefig(os.path.join(log_dir, 'lr_scheduler.jpg'), dpi=600, bbox_inches='tight')

def plot_loss(log_dir, train_loss_list, val_loss_list):
    plt.figure()
    plt.plot(train_loss_list, c='r', label='train loss', linewidth=2)
    plt.plot(val_loss_list, c='b', label='val loss', linewidth=2)
    plt.legend(loc='best')
    plt.xlabel('epoch', fontsize=10)
    plt.ylabel('loss', fontsize=10)
    plt.yscale("log")
    plt.savefig(os.path.join(log_dir, 'plot_loss.jpg'), dpi=600, bbox_inches='tight')