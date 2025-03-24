import os
import cv2
import lpips
import torch
import pandas as pd
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def calculate_metrics(hr_img, sr_img, lpips_model):
    psnr_value = psnr(hr_img, sr_img, data_range=255)
    ssim_value = ssim(hr_img, sr_img, channel_axis=-1, data_range=255, win_size=min(7, min(hr_img.shape[:2])))
    
    hr_tensor = transforms.ToTensor()(hr_img).unsqueeze(0).cuda()
    sr_tensor = transforms.ToTensor()(sr_img).unsqueeze(0).cuda()
    lpips_value = lpips_model(hr_tensor, sr_tensor).item()
    
    return psnr_value, ssim_value, lpips_value

def main():
    hr_dir = "/data1/jianglei/work/dataset/HoliSDiP/StableSR_testsets/DIV2K_V2_val/gt"
    sr_dir = "/data1/jianglei/work/HoliSDiP/result_230000/samples"
    save_path = "sr_evaluation_add_global_caption_230000.xlsx"
    
    lpips_model = lpips.LPIPS(net='alex').cuda()
    
    psnr_list, ssim_list, lpips_list = [], [], []
    
    filenames = sorted(os.listdir(hr_dir))
    
    for filename in filenames:
        hr_path = os.path.join(hr_dir, filename)
        sr_path = os.path.join(sr_dir, filename)
        
        if os.path.exists(sr_path):
            hr_img = load_image(hr_path)
            sr_img = load_image(sr_path)
            
            metrics = calculate_metrics(hr_img, sr_img, lpips_model)
            psnr_list.append(metrics[0])
            ssim_list.append(metrics[1])
            lpips_list.append(metrics[2])
            print(metrics[0])
    
    avg_metrics = {
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "LPIPS": np.mean(lpips_list)
    }
    
    df = pd.DataFrame([avg_metrics])
    df.to_excel(save_path, index=False)

if __name__ == "__main__":
    main()
