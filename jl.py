import torch
import cv2
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.projects.deeplab import add_deeplab_config
from Mask2Former.mask2former import add_maskformer2_config

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
image_path = "/data1/jianglei/work/dataset/HoliSDiP/StableSR_testsets/DIV2K_V2_val/gt/0801_pch_00002.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 进行全景分割
outputs = predictor(image)

# 确保 panoptic_seg 结果在 CPU 上
panoptic_seg, segments_info = outputs["panoptic_seg"]
panoptic_seg = panoptic_seg.to("cpu")

# 获取类别元数据
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# 提取所有类别标签
unique_labels = set()
for segment in segments_info:
    category_id = segment["category_id"]
    
    # 检查 metadata 是否包含 thing_classes 和 stuff_classes
    thing_classes = metadata.thing_classes if hasattr(metadata, "thing_classes") else []
    stuff_classes = metadata.stuff_classes if hasattr(metadata, "stuff_classes") else []
    
    # 确定标签
    if category_id < len(thing_classes):
        label = thing_classes[category_id]
    elif category_id < len(thing_classes) + len(stuff_classes):
        label = stuff_classes[category_id - len(thing_classes)]
    else:
        label = f"Unknown_{category_id}"  # 避免越界
    
    unique_labels.add(label)

# 输出所有检测到的类别
print("Detected Labels:", unique_labels)

# 可视化分割结果
v = Visualizer(image_rgb, metadata, scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info)

# 显示并保存结果
cv2.imwrite("output.jpg", cv2.cvtColor(out.get_image(), cv2.COLOR_RGB2BGR))