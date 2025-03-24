from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from Mask2Former.mask2former import add_maskformer2_config
from Mask2Former.train_net import Trainer
from detectron2.checkpoint import DetectionCheckpointer
from PIL import Image
import numpy as np
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# 配置 Mask2Former 实例分割模型
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("/data1/jianglei/work/HoliSDiP/preset/models/mask2former/instance-segmentation/config/maskformer2_swin_large_IN21k_384_bs16_160k.yaml")
cfg.MODEL.WEIGHTS = "/data1/jianglei/work/HoliSDiP/preset/models/mask2former/instance-segmentation/model_final_92dae9.pkl"
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False  # 关闭语义分割

# 加载模型
model = Trainer.build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval()

# 读取输入图片
image_path = "/data1/jianglei/work/dataset/dog_cat.jpg"
image = Image.open(image_path)
image_np = np.array(image)
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
images = [{'image': image_tensor}]

# 进行实例分割推理
labels = model(images)

# 获取实例分割结果
instances = labels[0]["instances"].to("cpu")

# 获取元数据（用于可视化）
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# 转换 OpenCV 格式
image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# 可视化实例分割结果
visualizer = Visualizer(image_cv2, metadata, scale=1.2)
out = visualizer.draw_instance_predictions(instances)

# 保存可视化结果
output_image = out.get_image()[:, :, ::-1]  # 转换回 BGR 格式
cv2.imwrite("output_instance_segmentation.png", output_image)

print("实例分割结果已保存为: output_instance_segmentation.png")
