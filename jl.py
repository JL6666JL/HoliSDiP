import pickle

# 替换为你的 .pkl 文件路径
pkl_file_path = '/data2/jianglei/Holidataset/FFHQ_LSDIR/lfhf_local_descriptions/FFHQ_00000_00000_descriptions.pkl'

# 读取 pkl 文件中的字典
with open(pkl_file_path, 'rb') as f:
    data_dict = pickle.load(f)

# 可选：打印字典的内容
print("字典内容：")
for key, value in data_dict.items():
    print(f"{key}: {value}")
