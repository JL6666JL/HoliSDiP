import torch
from transformers import AutoTokenizer, PretrainedConfig
import numpy as np
from tqdm.auto import tqdm
import json
import os

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

# 文件路径
captions_path = '/data1/jianglei/work/dataset/HoliSDiP/descriptions.json'
embeddings_path = '/data1/jianglei/work/dataset/HoliSDiP/descriptions_embeddings.npy'

# 加载caption
with open(captions_path, 'r') as file:
    captions = json.load(file)

# 模型和tokenizer
pretrained_model_name_or_path = "/data1/jianglei/work/HoliSDiP/preset/models/stable-diffusion-2-base"
revision = None

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False
)

text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)
text_encoder = text_encoder_cls.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
)

# 将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_encoder.to(device)

# 批量处理
batch_size = 512  # 根据GPU内存调整
filenames = list(captions.keys())
caption_list = list(captions.values())
# embeddings = {}
npy_num = 0

for i in tqdm(range(0, len(caption_list), batch_size), desc="Processing captions"):
    batch_captions = caption_list[i:i + batch_size]
    batch_filenames = filenames[i:i + batch_size]

    # Tokenize
    inputs = tokenizer(
        batch_captions,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True
    )

    # # 将输入数据移动到GPU
    # inputs = {key: value.to(device) for key, value in inputs.items()}

    inputs = inputs.to(device)

    # 生成embedding
    with torch.no_grad():
        outputs = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        batch_embeddings = outputs.last_hidden_state  

    # 将embedding移回CPU并保存结果
    batch_embeddings = batch_embeddings.cpu()
    for filename, embedding in zip(batch_filenames, batch_embeddings):
        # embeddings[filename] = embedding.numpy()
        directory, imagename = os.path.split(filename)
        imagename_without_ext = os.path.splitext(imagename)[0]
        npy_path = os.path.join(directory, f'{imagename_without_ext}.npy')
        np.save(npy_path, embedding)
        npy_num += 1

print(f'embedding 数量:{npy_num}')
print("所有npy文件已经存储完毕")
# # 保存embedding
# np.save(embeddings_path, embeddings)