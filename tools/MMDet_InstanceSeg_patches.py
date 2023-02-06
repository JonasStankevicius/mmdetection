import matplotlib.pyplot as plt
from PIL import Image
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmdet
import torch
import torchvision
import os.path as osp
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import inference_detector, show_result_pyplot

cfg = Config.fromfile(
    './configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco_patches.py')

# Modify dataset type and path
cfg.dataset_type = 'COCODataset'

data_path = '/mnt/f/RoadPatches/'
# val_data_path = data_path+'val_vox0.02_size15.4_step4_COCO/'
val_data_path = data_path+'val_vox0.02_size15.4_step7_COCO/'
train_data_path = data_path+'train_vox0.02_size15.4_step4_COCO/'

cfg.data.test.ann_file = val_data_path+'annotation_coco.json'
cfg.data.test.img_prefix = val_data_path
cfg.data.test.classes = ('patches',)

cfg.data.train.ann_file = train_data_path+'annotation_coco.json'
cfg.data.train.img_prefix = train_data_path
cfg.data.train.classes = ('patches',)

# cfg.data.train.ann_file = val_data_path+'annotation_coco.json'
# cfg.data.train.img_prefix = val_data_path
# cfg.data.train.classes = ('patches',)

cfg.data.val.ann_file = val_data_path+'annotation_coco.json'
cfg.data.val.img_prefix = val_data_path
cfg.data.val.classes = ('patches',)

cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu = 4

# modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# We can still the pre-trained Mask RCNN model to obtain a higher performance
# cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
cfg.load_from = 'patches_exps/epoch_11.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './patches_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10
cfg.max_epochs = 20

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = 'cuda'

# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    # dict(type='TensorboardLoggerHook'),
    dict(type='MMDetWandbHook', init_kwargs={
        'entity': "jonasstankevicius",
        'project': "PatchDetection"
    }),
]

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)


# img = mmcv.imread(val_data_path+'road_patches_val_0.jpg')
# model.cfg = cfg
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result)