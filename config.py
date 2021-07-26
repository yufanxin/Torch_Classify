import torch

#  --------------------------------------------------------------------------------------
# |model_prefix    |model_suffix                                                         |
# |--------------------------------------------------------------------------------------|
# |vgg             |11 13 16 19 bn11 bn13 bn16 bn19                                      |
# |--------------------------------------------------------------------------------------|
# |resnet          |18 34 50 101 152                                                     |
# |--------------------------------------------------------------------------------------|
# |resnext         |50-32x4d 101-32x8d                                                   |
# |--------------------------------------------------------------------------------------|
# |regnetx         |200mf 400mf 600mf 800mf 1.6gf 3.2gf 4.0gf 6.4gf 8.0gf 12gf 16gf 32gf |
# |--------------------------------------------------------------------------------------|
# |regnety         |200mf 400mf 600mf 800mf 1.6gf 3.2gf 4.0gf 6.4gf 8.0gf 12gf 16gf 32gf |
# |--------------------------------------------------------------------------------------|
# |mobilenetv2     |0.25, 0.5, 0.75, 1.0, 1.25, 1.5                                      |
# |--------------------------------------------------------------------------------------|
# |mobilenetv3     |small large                                                          |
# |--------------------------------------------------------------------------------------|
# |ghostnet        |0.5 1.0 1.3                                                          |
# |--------------------------------------------------------------------------------------|
# |efficientnetv1  |b0 b1 b2 b3 b4 b5 b6 b7                                              |
# |--------------------------------------------------------------------------------------|
# |efficientnetv2  |small medium large                                                   |
# |--------------------------------------------------------------------------------------|
# |shufflenetv2    |0.5 1.0 1.5 2.0                                                      |
# |--------------------------------------------------------------------------------------|
# |densenet        |121 161 169 201                                                      |
# |--------------------------------------------------------------------------------------|
# |xception        |299                                                                  |
# |--------------------------------------------------------------------------------------|
# |vit             |base-patch16 base-patch32 large-patch16 large-patch32 huge-patch14   |
#  --------------------------------------------------------------------------------------

configurations = {
    'cfg': dict(
        load_from=r"D:\torch_classify\weights\mobilenetv2\mobilenet_v2.pth",  # pretrain weight of imagenet
        model_prefix='mobilenetv2',  # above model_prefix
        model_suffix='1.0',  # above model_suffix
        img_path='data',  # the parent root where your train/val data are stored, not support test data
                          # -data
                          #    -train
                          #       -class_0
                          #          -1.jpg
                          #       -class_1
                          #       -...
                          #    -val
                          #       -class_0
                          #       -class_1
                          #       -...
        predict_img_path=r"imgs/02500_05.jpg",  # only use in predict.py
        log_root='logs',  # the root to log your train/val status
        mean=[0.485, 0.456, 0.406],  # [0.485, 0.456, 0.406] if use pretrain weight of imagenet else [0.5, 0.5, 0.5]
        std=[0.229, 0.224, 0.225],  # [0.229, 0.224, 0.225] if use pretrain weight of imagenet else [0.5, 0.5, 0.5]
        img_size=[112, 112],  # especially for efficientnetv1 b0->224, b1->240, b2->260, b3->300, b4->380, b5->456, b6->528, b7->600
                              # especially for xception 299
        num_classes=10,
        batch_size=64,
        epochs=200,
        device="cuda:0",  #  now only support single gpu or cpu, ['cuda:0', 'cpu']
        num_workers=0,  # 0 if run on windows else depend on cpu
        init_lr=0.01,
        warmup_epochs=0,  # linear warm up 5 epoch
        scheduler_type='warmup_cosine_lr',  # support: ['warmup_cosine_lr', 'warmup_step_lr']
        steps=[100, 150, 180],  # use steps if scheduler_type=='warmup_step_lr' else ignore, default mutiply 0.1 when epoch == step
        lr_scale=0.01,  # use lr_scale if scheduler_type=='warmup_cosine_lr' else ignore, cosine_lr: init_lr -> init_lr*lr_scale
        drop_last=False,  # whether drop the last batch to ensure consistent batch_norm statistics
        pin_memory=True,  # True if you memory big enough else False
        optimizer_type='sgd',  # support: ['sgd', 'adam', 'adamw', 'rmsprop']
        loss_type='CELoss',  # support: ['CELoss', 'LabelSmoothCELoss'] smoothing default=0.1
        use_apex=True  # use apex to train by mixed-precision
    ),
}
