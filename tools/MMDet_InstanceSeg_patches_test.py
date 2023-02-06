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
from mmcv.runner import build_runner
from mmcv.runner import load_checkpoint
import os

cfg = Config.fromfile(
    './configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco_patches.py')

# Modify dataset type and path
cfg.dataset_type = 'COCODataset'

data_path = '/mnt/f/RoadPatches/'
# val_data_path = data_path+'val_vox0.02_size15.4_step4_COCO/'
val_data_path = data_path+'val_vox0.02_size15.4_step7_COCO/'
output_data_path = data_path+'val_result/'

if not os.path.exists(output_data_path):
    os.mkdir(output_data_path)

cfg.data.samples_per_gpu = 4
cfg.data.workers_per_gpu = 4

# modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# Set up working dir to save files and logs.
cfg.work_dir = './patches_exps'

# cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
cfg.load_from = 'patches_exps/epoch_11.pth'

# Build the detector
model = build_detector(cfg.model)

# Add an attribute for visualization convenience
model.CLASSES = ('patches',)

# mmcv.runner.checkpoint.load_checkpoint(model, cfg.load_from, strict=True)
load_checkpoint(model, cfg.load_from, map_location='cuda:0')

model.cfg = cfg
model.eval() ##### !!!!!!!!! ###### Dont forget this shit

for file in os.listdir(val_data_path):
    if file.endswith('jpg'):
        
        file_path = osp.join(val_data_path, file)
        out_file_path = osp.join(output_data_path, file)
        
        img = mmcv.imread(file_path)
        result = inference_detector(model, img)
        show_result_pyplot(model, img, result, out_file=out_file_path) #,out_file="result.jpg")