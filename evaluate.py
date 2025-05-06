import os
import torch
import pyiqa
import cv2
import pandas as pd
import numpy as np
from torchvision import transforms
import argparse

def load_image(path):
    """加载图像并转换为 PyTorch 张量 (1, C, H, W)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    return img_tensor

def calculate_metrics(hr_img, sr_img, metrics_dict):
    """计算全参考（PSNR, SSIM, LPIPS）和无参考（MUSIQ, MANIQA, CLIPIQA）指标"""
    psnr_value = metrics_dict["psnr"](sr_img, hr_img).item()
    ssim_value = metrics_dict["ssim"](sr_img, hr_img).item()
    lpips_value = metrics_dict["lpips"](sr_img, hr_img).item()

    musiq_value = metrics_dict["musiq"](sr_img).item()
    maniqa_value = metrics_dict["maniqa"](sr_img).item()
    clipiqa_value = metrics_dict["clipiqa"](sr_img).item()

    return psnr_value, ssim_value, lpips_value, musiq_value, maniqa_value, clipiqa_value

def main(args):

    hr_dir = "/data1/jianglei/work/dataset/HoliSDiP/StableSR_testsets/DIV2K_V2_val/gt"
    # sr_dir = "/data1/jianglei/work/HoliSDiP/results/add_global_local_des_290000/samples"
    sr_dir = os.path.join(args.sr_dir,"samples")
    save_path = "/data2/jianglei/HoliSDiP/results/add_global_local_des~LGQS__keeppos_results.xlsx"
    # save_path = "sr_evaluation_add_global_local_des_290000.xlsx"

    # 创建 `pyiqa` 评估器
    metrics_dict = {
        "psnr": pyiqa.create_metric('psnr'),
        "ssim": pyiqa.create_metric('ssim'),
        "lpips": pyiqa.create_metric('lpips'),
        "musiq": pyiqa.create_metric('musiq'),
        "maniqa": pyiqa.create_metric('maniqa'),
        "clipiqa": pyiqa.create_metric('clipiqa'),
    }

    psnr_list, ssim_list, lpips_list = [], [], []
    musiq_list, maniqa_list, clipiqa_list = [], [], []

    filenames = sorted(os.listdir(hr_dir))

    for filename in filenames:
        hr_path = os.path.join(hr_dir, filename)
        sr_path = os.path.join(sr_dir, filename)

        if os.path.exists(sr_path):
            hr_img = load_image(hr_path)
            sr_img = load_image(sr_path)

            metrics = calculate_metrics(hr_img, sr_img, metrics_dict)
            psnr_list.append(metrics[0])
            ssim_list.append(metrics[1])
            lpips_list.append(metrics[2])
            musiq_list.append(metrics[3])
            maniqa_list.append(metrics[4])
            clipiqa_list.append(metrics[5])

            print(f"{filename} - PSNR: {metrics[0]:.4f}, SSIM: {metrics[1]:.4f}, LPIPS: {metrics[2]:.4f}, "
                  f"MUSIQ: {metrics[3]:.4f}, MANIQA: {metrics[4]:.4f}, CLIPIQA: {metrics[5]:.4f}")
        else:
            print("SR路径不存在!")

    avg_metrics = {
        "steps" : args.steps,
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "LPIPS": np.mean(lpips_list),
        "MUSIQ": np.mean(musiq_list),
        "MANIQA": np.mean(maniqa_list),
        "CLIPIQA": np.mean(clipiqa_list),
    }

    df = pd.DataFrame([avg_metrics])

    if os.path.exists(save_path):
        old_df = pd.read_excel(save_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_excel(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_dir", type=str, default=None)
    parser.add_argument("--steps", type=str, default=None)
    args = parser.parse_args()
    main(args)
