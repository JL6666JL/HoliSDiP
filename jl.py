import numpy as np
import os

def load_embedding(image_path):
    directory, filename = os.path.split(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    npy_path = os.path.join(directory, f'{filename_without_ext}.npy')

    if os.path.exists(npy_path):
        return np.load(npy_path)
    else:
        raise FileNotFoundError(f'Embedding file not found: {npy_path}')

# 示例
image_path = "/data2/jianglei/dataset/HoliSDiP/LSDIR/0067000/0066418.png"
embedding = load_embedding(image_path)
print(f'Loaded embedding for {image_path}, shape: {embedding.shape}')
