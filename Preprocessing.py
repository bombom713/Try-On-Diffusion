import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from PIL import Image
import torchvision.transforms as transforms

def sp_sg(data):
    segmentations = []
    if isinstance(data, list):
        for item in data:
            segmentations.append(torch.tensor(item).float())
    else:
        for key in data:
            if key == 'segmentation':
                if isinstance(data[key], list):
                    for item in data[key]:
                        segmentations.append(torch.tensor(item).float())
                else:
                    segmentations.append(torch.tensor(data[key]).float())
            elif isinstance(data[key], dict):
                segmentations.extend(sp_sg(data[key]))
    return segmentations

spjson = "./Model-Parse/1008_A001_000.json"
sgjson = "./Item-Parse/1008002_F.json"

with open(spjson, 'r') as f:
    data_sp = json.load(f)
Sp = sp_sg(data_sp)[0]  # person human parsing map

with open(sgjson, 'r') as f:
    data_sg = json.load(f)
Sg = sp_sg(data_sg)[0]  # garment human parsing map

# print(Sp, Sg)

def jp_jg(data):
    landmarks = []
    if isinstance(data, list):
        landmarks.append(torch.tensor(data).float())
    else:
        for key in data:
            if key == 'landmarks':
                landmarks.append(torch.tensor(data[key]).float())
            elif isinstance(data[key], dict):
                landmarks.extend(jp_jg(data[key]))
    return landmarks

jpjson = "./Model-Pose/1008_A001_000.json"
jgjson = "./Item-Pose/1008002_F.json"

with open(jpjson, 'r') as f:
    data_jp = json.load(f)
Jp = jp_jg(data_jp)[0]  # person pose keypoints

with open(jgjson, 'r') as f:
    data_jg = json.load(f)
Jg = jp_jg(data_jg)[0]  # garment pose keypoints

# print(Jp)
# print(Jg)

def Ip_Ic(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image)

image_path_person = './Model-Image/1008_A001_000.jpg'
image_path_garment = './Item-Image/1008002_F.jpg'

# 이미지 로드
Ip = Ip_Ic(image_path_person)
Ic = Ip_Ic(image_path_garment)

# print(Ip)
# print(Ic)

# Assuming Ip and Ic are given
Sp = F.interpolate(Sp.unsqueeze(0).unsqueeze(0), size=(Ip.size(1), Ip.size(2)), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
Sg = F.interpolate(Sg.unsqueeze(0).unsqueeze(0), size=(Ic.size(1), Ic.size(2)), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

Ia = preprocess_person_image(Ip, Sp, Jp)
Ic = Ic * Sg  # Segment out the garment using the parsing map

# Normalize pose keypoints to the range of [0, 1]
Jp = (Jp - Jp.min()) / (Jp.max() - Jp.min())
Jg = (Jg - Jg.min()) / (Jg.max() - Jg.min())

# Conditional inputs for try-on
ctryon = (Ia, Jp, Ic, Jg)

print(ctryon)
