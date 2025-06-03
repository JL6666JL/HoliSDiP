import pickle

# 替换为你的 .pkl 文件路径
pkl_file_path = '/data1/jianglei/work/dataset/test/des_lq/test_imagelq_0807_pch_00014_descriptions.pkl'

# 读取 pkl 文件中的字典
with open(pkl_file_path, 'rb') as f:
    data_dict = pickle.load(f)

# # 可选：打印字典的内容
# print("字典内容：")
# for key, value in data_dict.items():
#     print(f"{key}: {value}")

# print()

print(type(data_dict['panoptic_seg']))
print(len(data_dict['panoptic_seg']))
print(len(data_dict['panoptic_seg'][0]))