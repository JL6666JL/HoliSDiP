import numpy as np
import os
from tqdm import tqdm

# 读取.npy文件
embeddings_path = '/data2/jianglei/dataset/HoliSDiP/captions_embeddings.npy'
embeddings = np.load(embeddings_path, allow_pickle=True).item()

# 保存每个caption的embedding到对应图片路径
for filepath, embedding in tqdm(embeddings.items(), desc='Saving embeddings'):
    directory, filename = os.path.split(filepath)
    filename_without_ext = os.path.splitext(filename)[0]
    npy_path = os.path.join(directory, f'{filename_without_ext}.npy')
    np.save(npy_path, embedding)

print(f'Saved {len(embeddings)} embeddings to their respective image directories.')