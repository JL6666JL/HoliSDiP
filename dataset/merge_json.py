# # 合并各个descriptions的json文件

# import json

# # 定义输出文件
# output_file = 'combined.json'
# combined_data = {}

# # 读取所有JSON文件
# json_files = ['/data1/jianglei/work/dataset/HoliSDiP/descriptions_0to25000.json',
#               '/data1/jianglei/work/dataset/HoliSDiP/descriptions_25000to50000.json',
#               '/data1/jianglei/work/dataset/HoliSDiP/descriptions_50000to75000.json',
#               '/data1/jianglei/work/dataset/HoliSDiP/descriptions_75000to94991.json']  # 替换为你的文件列表

# for file in json_files:
#     with open(file, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line.strip())  # 解析每一行的JSON
#             combined_data.update(data)  # 合并到大的JSON对象中

# # 写入合并后的文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     json.dump(combined_data, f, indent=4, ensure_ascii=False)


import json

def merge_json_files(file_list, output_file):
    combined_data = {}
    duplicate_keys = set()  # 用于记录重复的key

    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())  # 解析每一行的JSON
                key = list(data.keys())[0]  # 获取当前行的key

                # 检查key是否已经存在
                if key in combined_data:
                    duplicate_keys.add(key)  # 记录重复的key
                else:
                    combined_data[key] = data[key]  # 添加到合并的数据中

    # 如果发现重复key，抛出错误
    if duplicate_keys:
        raise ValueError(f"发现重复的key: {duplicate_keys}")

    # 写入合并后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)

    print(f"合并完成，结果已保存到 {output_file}")

# 示例调用
if __name__ == "__main__":
    # 读取所有JSON文件
    json_files = ['/data1/jianglei/work/dataset/HoliSDiP/descriptions_0to25000.json',
                '/data1/jianglei/work/dataset/HoliSDiP/descriptions_25000to50000.json',
                '/data1/jianglei/work/dataset/HoliSDiP/descriptions_50000to75000.json',
                '/data1/jianglei/work/dataset/HoliSDiP/descriptions_75000to94991.json']  # 替换为你的文件列表
    output_file = 'combined.json'  # 合并后的输出文件

    try:
        merge_json_files(json_files, output_file)
    except ValueError as e:
        print(f"合并失败: {e}")