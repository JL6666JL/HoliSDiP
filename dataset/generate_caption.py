import pandas as pd
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import os
import json
from tqdm.auto import tqdm

input_path = ['/data2/jianglei/dataset/HoliSDiP/FFHQ','/data2/jianglei/dataset/HoliSDiP/LSDIR']  # path for HR
# input_path = ['/data2/jianglei/dataset/test']  # path for HR


result_name = f"/data2/jianglei/dataset/HoliSDiP/captions.json"

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
torch_dtype=torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = torch.nn.DataParallel(model)
model.to(device,dtype=torch.float16)

img_num = 0
for dataset_path in input_path:
    for imgdir in os.scandir(dataset_path):
        if imgdir.is_dir():
            img_num += len(os.listdir(os.path.join(dataset_path,imgdir)))
print(f'total number of image is {img_num} ')

progress_bar = tqdm(
    range(0, img_num),
    initial=0,
    desc="Steps",
)

captions = {}

for dataset_path in input_path:
    for imgdir in os.scandir(dataset_path):
        if imgdir.is_dir():
            for img_name in os.listdir(os.path.join(dataset_path,imgdir)):
                img_path = os.path.join(dataset_path,imgdir,img_name)
                
                # load image
                image = Image.open(img_path).convert('RGB')

                prompt = "Question: Please describe the contents in the photo in details. Answer:"
                inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                generated_text = generated_text.lower().replace('.', ',').rstrip(',')
                caption = generated_text

                captions[img_path] = caption
                progress_bar.update(1)


with open(result_name, 'w') as file:
    json.dump(captions , file , indent=4)



