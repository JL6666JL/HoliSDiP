import torch

# 加载 optimizer.bin 文件
optimizer_state = torch.load("/data1/jianglei/work/HoliSDiP_fuxian/experiments/auth/optimizer.bin", map_location="cpu")

# 检查优化器状态字典的键名
print(optimizer_state.keys())  # 输出所有键，寻找 'step' 或 'state'

# 如果使用 Adam/AdamW，步数通常存储在优化器的 'state' 字段中
if 'state' in optimizer_state:
    for param_id in optimizer_state['state']:
        if 'step' in optimizer_state['state'][param_id]:
            print(f"Training step: {optimizer_state['state'][param_id]['step']}")
            break