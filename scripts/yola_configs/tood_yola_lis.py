# YOLA（TOOD）在 LIS 上微调：继承官方 ExDark 配置，只改数据集和类别数，其余（模型、数据增强、
# 学习率、轮数）全部沿用官方设置。从官方发布的 ExDark 权重出发。
# 由 scripts\train_yola_lis.py 读取；路径按本文件所在位置写成相对路径。
_base_ = ['../../code/YOLA/configs/tood/tood_yola_exdark.py']

load_from = '../../weights/yola/tood_exdark/753.pth'   # 官方 ExDark 权重；分类层类别数不同，会被跳过重新初始化

# backbone 的 ImageNet 初始化用不上（马上会被 load_from 覆盖），关掉以免多下载一次 ResNet-50
model = dict(backbone=dict(init_cfg=None), bbox_head=dict(num_classes=8))

dataset_type = 'CocoDataset'
data_root = '../../data/LIS/'
metainfo = dict(classes=('bicycle', 'car', 'motorbike', 'bus', 'bottle', 'chair', 'diningtable', 'tvmonitor'))

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/coco_dark.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline={{_base_.train_pipeline}}))
test_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/coco_dark.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline={{_base_.test_pipeline}}))

# 训练中不做验证：固定用最后一个 epoch，测试集只在训练结束后用一次
val_dataloader = None
val_evaluator = None
val_cfg = None
test_evaluator = dict(_delete_=True, type='CocoMetric', ann_file=data_root + 'test/coco_dark.json', metric='bbox')

# 每个 checkpoint 约 260 MB，只留最近 2 个
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2))
