import cv2
import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
import math

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor

from PIL import Image
import pickle
from torchvision.utils import save_image

class DataLoader(Dataset):
    def __init__(self, opt, fix_size=512): 
        
        self.opt = opt['kernel_info']
        self.image_root = opt['gt_path']
        self.fix_size = fix_size
        exts = ['*.jpg', '*.png']
        
        self.image_list = []
        for image_root in self.image_root:
            for ext in exts:
                image_list = sorted(glob.glob(os.path.join(image_root, ext)))
                self.image_list += image_list
                # if add lsdir dataset
                image_list = sorted(glob.glob(os.path.join(image_root, '00*', ext)))
                self.image_list += image_list

        self.img_preproc = transforms.Compose([
            transforms.RandomCrop(fix_size),
            transforms.ToTensor(),
        ])

        # blur settings for the first degradation
        self.blur_kernel_size = self.opt['blur_kernel_size']
        self.kernel_list = self.opt['kernel_list']
        self.kernel_prob = self.opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = self.opt['blur_sigma']
        self.betag_range = self.opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = self.opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = self.opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = self.opt['blur_kernel_size2']
        self.kernel_list2 = self.opt['kernel_list2']
        self.kernel_prob2 = self.opt['kernel_prob2']
        self.blur_sigma2 = self.opt['blur_sigma2']
        self.betag_range2 = self.opt['betag_range2']
        self.betap_range2 = self.opt['betap_range2']
        self.sinc_prob2 = self.opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = self.opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        print(f'The dataset length: {len(self.image_list)}')

        # self.seg_description_emb_path = "/data2/jianglei/dataset/HoliSDiP/local_descriptions"
        self.seg_description_emb_path = opt['seg_description_emb_path']


    def load_caption_embedding(self,image_path):
        directory, filename = os.path.split(image_path)
        filename_without_ext = os.path.splitext(filename)[0]
        npy_path = os.path.join(directory, f'{filename_without_ext}.npy')

        if os.path.exists(npy_path):
            return np.load(npy_path)
        else:
            raise FileNotFoundError(f'Embedding file not found: {npy_path}')
    
    def load_seg_description_emb(self,image_path):
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        parent_dir = os.path.dirname(image_path)
        folder_name = os.path.basename(parent_dir) 

        dataset_name = os.path.basename(os.path.dirname(parent_dir))

        file_name = f"{dataset_name}_{folder_name}_{image_name}_descriptions.pkl"
        with open(os.path.join(self.seg_description_emb_path,file_name), 'rb') as f:
            local_descriptions = pickle.load(f)
        return local_descriptions["panoptic_seg"], local_descriptions["seg_emb_dict"]

    def synchronized_crop_to_tensor(self, image, panoptic_seg, crop_size):
        """
        对 PIL.Image 和 numpy.ndarray 同步随机裁剪，并转换为 Tensor
        Args:
            image: PIL.Image (RGB)
            panoptic_seg: numpy.ndarray (H x W)
            crop_size: 目标裁剪尺寸 (h, w)
        Returns:
            image_tensor: torch.Tensor (C x H x W)
            panoptic_tensor: torch.Tensor (H x W)
        """
        # 1. 生成随机裁剪位置
        width, height = image.size
        h = crop_size
        w = crop_size
        i = np.random.randint(0, height - h + 1) if height > h else 0
        j = np.random.randint(0, width - w + 1) if width > w else 0

        # 2. 裁剪图像和分割图
        image_cropped = transforms.functional.crop(image, i, j, h, w)  # PIL.Image
        panoptic_cropped = panoptic_seg[i:i+h, j:j+w]                   # numpy.ndarray

        # 3. 统一转换为 Tensor
        image_tensor = transforms.ToTensor()(image_cropped)  # 自动归一化到 [0,1] (C x H x W)
        panoptic_tensor = torch.from_numpy(panoptic_cropped) # 保持原始值 (H x W)

        return image_tensor, panoptic_tensor

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        panoptic_seg, seg_emb_dict = self.load_seg_description_emb(self.image_list[index])
        # image = self.img_preproc(image)
        image, panoptic_seg = self.synchronized_crop_to_tensor(image, panoptic_seg, self.fix_size)

        # # 检验是否裁剪的是同样的位置
        # # 定义颜色映射（0-10 分别对应 RGB 颜色）
        # color_map = {
        #     0: [0, 0, 0],       # 黑色
        #     1: [255, 0, 0],     # 红色
        #     2: [0, 0, 255],     # 蓝色
        #     3: [0, 255, 0],     # 绿色
        #     4: [255, 255, 0],   # 黄色
        #     5: [128, 0, 128],   # 紫色
        #     6: [255, 165, 0],   # 橙色
        #     7: [0, 255, 255],   # 青色
        #     8: [255, 0, 255],   # 品红
        #     9: [0, 255, 0],     # 绿色（可自定义）
        #     10: [255, 255, 255] # 白色
        # }
        # h,w = panoptic_seg.shape
        # rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        # for val in color_map:
        #     rgb_image[panoptic_seg == val] = color_map[val]
        # # print(image.shape,panoptic_seg.shape)
        # # print(panoptic_seg)
        # cv2.imwrite("crop_mask.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        # save_image(image,"crop_image.png")
        # input("11")
        
        image_caption_emb = self.load_caption_embedding(self.image_list[index])
    
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        # self.image_list里面存储的好像是HR图像，为什么被叫做lq_path。不过这个参数后面似乎也没用上
        return_d = {'gt': image, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 
                    'lq_path': self.image_list[index], 'caption_emb' : image_caption_emb,
                    "panoptic_seg": panoptic_seg, "seg_emb_dict": seg_emb_dict}
        return return_d
        

    def __len__(self):
        return len(self.image_list)
        
