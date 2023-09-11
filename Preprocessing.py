import torch
from PIL import Image
import torchvision.transforms as transforms
import json
import torch.nn.functional as F

def get_pre():
    def load_image(image_path, desired_size=(256, 256)):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(desired_size),
            transforms.ToTensor()
        ])
        return transform(image).float()

    def sp_sg(data):
        segmentations = []
        if isinstance(data, list):
            for item in data:
                segmentations.append(torch.tensor(item, dtype=torch.float32))
        else:
            for key in data:
                if key == 'segmentation':
                    if isinstance(data[key], list):
                        for item in data[key]:
                            segmentations.append(torch.tensor(item, dtype=torch.float32))
                    else:
                        segmentations.append(torch.tensor(data[key], dtype=torch.float32))
                elif isinstance(data[key], dict):
                    segmentations.extend(sp_sg(data[key]))
        return segmentations

    def jp_jg(data):
        landmarks = []
        if isinstance(data, list):
            landmarks.append(torch.tensor(data, dtype=torch.float32))
        else:
            for key in data:
                if key == 'landmarks':
                    landmarks.append(torch.tensor(data[key], dtype=torch.float32))
                elif isinstance(data[key], dict):
                    landmarks.extend(jp_jg(data[key]))
        return landmarks

    def generate_Ia(Ip, Sp):
        # Ensure both tensors are of type float32
        assert Ip.dtype == torch.float32, "Ip tensor must be of type float32."
        assert Sp.dtype == torch.float32, "Sp tensor must be of type float32."

        # Resize Sp to match the size of Ip
        Sp_resized = F.interpolate(Sp.unsqueeze(0).unsqueeze(0), size=Ip.shape[1:], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)

        masked_person = Ip * (1 - Sp_resized)
        return masked_person


    image_path_person = './Model-Image/1008_A001_000.jpg'
    image_path_garment = './Item-Image/1008002_F.jpg'

    Ip = load_image(image_path_person)
    Ic = load_image(image_path_garment)

    spjson = "./Model-Parse/1008_A001_000.json"
    sgjson = "./Item-Parse/1008002_F.json"

    with open(spjson, 'r') as f:
        data_sp = json.load(f)
    Sp = sp_sg(data_sp)[0]  # person human parsing map

    with open(sgjson, 'r') as f:
        data_sg = json.load(f)
    Sg = sp_sg(data_sg)[0]  # garment human parsing map

    jpjson = "./Model-Pose/1008_A001_000.json"
    jgjson = "./Item-Pose/1008002_F.json"

    with open(jpjson, 'r') as f:
        data_jp = json.load(f)
    Jp = jp_jg(data_jp)[0]  # human pose keypoints

    with open(jgjson, 'r') as f:
        data_jg = json.load(f)
    Jg = jp_jg(data_jg)[0]  # garment pose keypoints

    Ia = generate_Ia(Ip, Sp)
   
    # Resize Sg to match the size of Ic
    Sg_resized = F.interpolate(Sg.unsqueeze(0).unsqueeze(0), size=Ic.shape[1:], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)

    # Add channel dimension to Sg_resized
    Sg_resized = Sg_resized.unsqueeze(0).expand_as(Ic)

    assert Ic.size() == Sg_resized.size(), "The sizes of Ic and Sg must be the same."
    Ic = Ic * Sg_resized  # Segment out the garment using the resized parsing map
    
    z_t = torch.randn_like(Ia)
    ctryon = (Ia, Jp, Ic, Jg)

    return ctryon, z_t, Ip
