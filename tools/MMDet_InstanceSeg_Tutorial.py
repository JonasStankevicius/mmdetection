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
import PIL.ImageDraw as ImageDraw

def convert_balloon_to_coco(ann_file, out_file, image_prefix, visualize=False):
    
    if osp.exists(out_file):
        print(f"{out_file} exists!")
        return
    
    data_infos = mmcv.load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            
            if visualize:
                img = mmcv.imread(img_path)
                # plt.figure(figsize=(15, 10))
                # plt.imshow(mmcv.bgr2rgb(img))
                # plt.show()
                image = Image.fromarray(img, mode='RGB') # IMAGE SHOULD BE BGR
                draw = ImageDraw.Draw(image)
                draw.polygon((poly), fill=200)
                image.save('poly.jpg')
            
            poly = [p for x in poly for p in x] # flatten poly coordinates

            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 'balloon'}])
    mmcv.dump(coco_format_json, out_file)

print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
print(mmdet.__version__)

# Check mmcv installation
print(get_compiling_cuda_version())
print(get_compiler_version())

# downlaod weights
# !mkdir checkpoints
# !wget -c https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth -O checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth

# Let's take a look at the dataset image

# img = mmcv.imread('balloon/train/10464445726_6f1e3bbe6a_k.jpg')
# plt.figure(figsize=(15, 10))
# plt.imshow(mmcv.bgr2rgb(img))
# plt.show()
# Image.fromarray(img, mode='RGB').show()

# Check the label of a single image
annotation = mmcv.load('balloon/train/via_region_data.json')
# The annotation is a dict, and its values looks like the following
print(annotation['34020010494_e5cb88e1c4_k.jpg1115004'])

convert_balloon_to_coco(
    'balloon/train/via_region_data.json',
    'balloon/train/annotation_coco.json',
    'balloon/train/')
convert_balloon_to_coco(
    'balloon/val/via_region_data.json',
    'balloon/val/annotation_coco.json',
    'balloon/val/')

cfg = Config.fromfile(
    './configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py')

# Modify dataset type and path
cfg.dataset_type = 'COCODataset'

cfg.data.test.ann_file = 'balloon/val/annotation_coco.json'
cfg.data.test.img_prefix = 'balloon/val/'
cfg.data.test.classes = ('balloon',)

cfg.data.train.ann_file = 'balloon/train/annotation_coco.json'
cfg.data.train.img_prefix = 'balloon/train/'
cfg.data.train.classes = ('balloon',)


cfg.data.val.ann_file = 'balloon/val/annotation_coco.json'
cfg.data.val.img_prefix = 'balloon/val/'
cfg.data.val.classes = ('balloon',)

cfg.data.samples_per_gpu = 2
cfg.data.workers_per_gpu = 0

# modify num classes of the model in box head and mask head
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1

# We can still the pre-trained Mask RCNN model to obtain a higher performance
cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 12

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
        'project': "BalloonDetection"
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


img = mmcv.imread('balloon/train/7178882742_f090f3ce56_k.jpg')
model.cfg = cfg
result = inference_detector(model, img)
show_result_pyplot(model, img, result)