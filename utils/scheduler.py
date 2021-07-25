from bisect import bisect_right
import math
import torch.optim.lr_scheduler as lr_scheduler

def get_lr(optimizer):
        return optimizer.param_groups[0]["lr"]

def warmup_cosine_lr(epochs, lr_scale=0.0001, warmup_epochs=5):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    return lambda epoch: (1+math.cos(math.pi*(epoch-warmup_epochs)/(epochs-warmup_epochs)))/2*(1-lr_scale)+lr_scale \
                  if epoch>warmup_epochs else epoch/warmup_epochs

def warmup_step_lr(steps, warmup_epochs=5):
    return lambda epoch: epoch / warmup_epochs if epoch <= warmup_epochs else 0.1**len([m for m in steps if m <= epoch])

def create_scheduler(scheduler_type, optimizer, epochs, lr_scale, steps, warmup_epochs=20):

    if scheduler_type == 'warmup_cosine_lr':
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lr(epochs, lr_scale, warmup_epochs))
        # return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lr_scale) + lr_scale)
    elif scheduler_type == 'warmup_step_lr':
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_step_lr(steps, warmup_epochs=warmup_epochs))
    else:
        raise ValueError('Unsupported scheduler_type - `{}`, '
                         'Use warmup_cosine_lr, warmup_step_lr'.format(scheduler_type))