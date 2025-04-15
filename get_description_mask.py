import torch
import cv2
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.projects.deeplab import add_deeplab_config
from Mask2Former.mask2former import add_maskformer2_config

import os
import numpy as np

# 设置配置
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("/data1/jianglei/work/HoliSDiP/preset/models/mask2former/panoptic-segmentation/config/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值
cfg.MODEL.WEIGHTS = "/data1/jianglei/work/HoliSDiP/preset/models/mask2former/panoptic-segmentation/model_final_f07440.pkl"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
predictor = DefaultPredictor(cfg)

# 读取图像
image_path = "/data1/jianglei/work/dataset/dog_cat.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 进行全景分割
outputs = predictor(image)

# 创建输出文件夹
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 确保 panoptic_seg 结果在 CPU 上
panoptic_seg, segments_info = outputs["panoptic_seg"]
panoptic_seg = panoptic_seg.to("cpu").numpy()

# 遍历分割结果并保存每个物体
for segment in segments_info:
    category_id = segment["category_id"]
    isthing = segment["isthing"]  # 是否为thing类
    segment_id = segment["id"]
    
    # 创建 mask
    mask = (panoptic_seg == segment_id).astype(np.uint8) * 255
    
    # 提取物体区域
    segmented_image = image.copy()
    segmented_image[mask == 0] = (0, 0, 0)  # 将非物体区域置黑
    
    # 保存图像
    filename = f"{output_dir}/object_{segment_id}_class_{category_id}.png"
    cv2.imwrite(filename, segmented_image)
    print(f"Saved: {filename}")


