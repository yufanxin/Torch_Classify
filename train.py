import json

from models.create_models import create_model
import torch
from torchvision import transforms, datasets
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.confusion import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from utils.scheduler import get_lr, create_scheduler
from utils.optimizer import create_optimizer
from utils.plots import plot_datasets, plot_txt, plot_lr_scheduler
from utils.loss import create_loss
from config import configurations
import numpy as np

cfg = configurations['cfg']
load_from = cfg['load_from']
img_path = cfg['img_path']
mean = cfg['mean']
std = cfg['std']
img_size = cfg['img_size']
num_classes = cfg['num_classes']
batch_size = cfg['batch_size']
epochs = cfg['epochs']
nw = cfg['num_workers']
device = cfg['device']
scheduler_type = cfg['scheduler_type']
model_prefix = cfg['model_prefix']
model_suffix = cfg['model_suffix']
init_lr = cfg['init_lr']
lr_scale = cfg['lr_scale']
drop_last = cfg['drop_last']
pin_memory = cfg['pin_memory']
optimizer_type = cfg['optimizer_type']
log_root = cfg['log_root']
steps = cfg['steps']
warmup_epochs = cfg['warmup_epochs']
loss_type = cfg['loss_type']
use_apex = cfg['use_apex']
model_name = model_prefix + '_' + model_suffix
log_dir = os.path.join(log_root, model_name)
os.makedirs(log_dir, exist_ok=True)
plot_datasets(img_path, log_dir)
results_file = os.path.join(log_dir, 'results.txt')
train_root = os.path.join(img_path, "train")
val_root = os.path.join(img_path, "val")
tb_writer = SummaryWriter(log_dir=log_dir)

print('[INFO] Using Model:{} Epoch:{} BatchSize:{} LossType:{} '
      'OptimizerType:{} SchedulerType:{}...'.format(model_name, epochs, batch_size,
                                                 loss_type, optimizer_type, scheduler_type))
print('[INFO] Logs will be saved in {}...'.format(log_dir))

class_index = ' '.join([str(i) for i in np.arange(num_classes)])
with open(results_file, 'w') as f:
    f.write('epoch ' + 'accuracy ' + 'precision ' + 'recall ' + 'F1-score ' + \
            class_index + ' ' + class_index + ' ' + class_index)

print("[INFO] Using {} device...".format(device))

data_transform = {"train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)]),
                  "val": transforms.Compose([transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])}

train_dataset = datasets.ImageFolder(root=train_root, transform=data_transform["train"])
train_num = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, drop_last=drop_last,
                                           shuffle=True, pin_memory=pin_memory,
                                           num_workers=nw)

validate_dataset = datasets.ImageFolder(root=val_root, transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, drop_last=drop_last,
                                              shuffle=False, pin_memory=pin_memory,
                                              num_workers=nw)
print('[INFO] Load Image From {}...'.format(img_path))

# write dict into json file
labels_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in labels_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open(os.path.join(log_dir, 'class_indices.json'), 'w') as json_file:
    json_file.write(json_str)
labels_name = list(cla_dict.values())

print('[INFO] {} to train, {} to val, total {} classes...'.format(train_num, val_num, num_classes))
net = create_model(model_name=model_name, num_classes=num_classes).to(device)

if load_from != "":
    print('[INFO] Load Weight From {}...'.format(load_from))
    if os.path.exists(load_from):
        load_weights_dict = {k: v for k, v in torch.load(load_from).items() if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        raise FileNotFoundError("[INFO] not found weights file: {}...".format(load_from))
print('[INFO] Successfully Load Weight From {}...'.format(load_from))
loss_function = create_loss(loss_type)

optimizer = create_optimizer(optimizer_type, net, init_lr)
if use_apex:
    from apex import amp
    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
scheduler = create_scheduler(scheduler_type, optimizer, epochs, lr_scale, steps, warmup_epochs)
plot_lr_scheduler(optimizer, scheduler, epochs, log_dir, scheduler_type)
best_acc = 0.0
train_steps = len(train_loader)
print('[INFO] Start Training...')

for epoch in range(epochs):
    net.train()
    train_per_epoch_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # print statistics
        train_per_epoch_loss += scaled_loss.item() if use_apex else loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f} lr:{:.6f}".format(epoch + 1, epochs, loss, get_lr(optimizer))

    train_per_epoch_loss = train_per_epoch_loss / train_steps

    # print('=' * 60)
    # validate
    net.eval()
    confusion = ConfusionMatrix(num_classes=num_classes)
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            confusion.update(val_labels, outputs, predict_y)
            confusion.acc_p_r_f1()
            val_bar.desc = "val epoch[{}/{}] Acc: {:.3f} P: {:.3f} R: {:.3f} F1: {:.3f}".format(epoch + 1, epochs,
                                                                                                confusion.mean_val_accuracy,
                                                                                                confusion.mean_precision,
                                                                                                confusion.mean_recall,
                                                                                                confusion.mean_F1)
    scheduler.step()

    tb_writer.add_scalar('train_loss', train_per_epoch_loss, epoch + 1)
    tb_writer.add_scalar('val_accuracy', confusion.mean_val_accuracy, epoch + 1)
    tb_writer.add_scalar('val_precision', confusion.mean_precision, epoch + 1)
    tb_writer.add_scalar('val_recall', confusion.mean_recall, epoch + 1)
    tb_writer.add_scalar('val_F1', confusion.mean_F1, epoch + 1)

    confusion.save(results_file, epoch + 1)

    if confusion.mean_val_accuracy > best_acc:
        best_acc = confusion.mean_val_accuracy
        torch.save(net.state_dict(), log_dir + '/best.pth')



torch.save(net.state_dict(), log_dir + '/last.pth')
plot_txt(log_dir, num_classes, labels_name)
print('[INFO] Results will be saved in {}...'.format(log_dir))
print('[INFO] Finished Training...')
