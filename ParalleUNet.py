import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from PIL import Image
import torchvision.transforms as transforms

class PoseEmbedding(nn.Module):
    def __init__(self, pose_dim, embedding_dim):
        super(PoseEmbedding, self).__init__()
        self.fc = nn.Linear(pose_dim, embedding_dim)
        
    def forward(self, pose):
        return self.fc(pose)

def preprocess_person_image(Ip, Sp, Jp):
    # 여러 세그멘테이션을 합칩니다.
    combined_segmentation = torch.sum(torch.stack(Sp), dim=0)
    
    # Mask out the whole bounding box area of the foreground person
    masked_person = Ip * (1 - combined_segmentation)
    
    # Copy-paste the head, hands, and lower body part on top of it
    # Assuming predefined masks for head, hands, and lower body based on Sp and Jp
    head_mask, hands_mask, lower_body_mask = generate_masks(combined_segmentation, Jp)
    clothing_agnostic = masked_person + Ip * head_mask + Ip * hands_mask + Ip * lower_body_mask

    return clothing_agnostic

def generate_masks(Sp, Jp):
    # This function should generate masks for head, hands, and lower body based on Sp and Jp
    # For simplicity, we're returning dummy masks. This needs to be implemented properly.
    head_mask = torch.zeros_like(Sp)
    hands_mask = torch.zeros_like(Sp)
    lower_body_mask = torch.zeros_like(Sp)
    return head_mask, hands_mask, lower_body_mask

person_pose = load_pose_from_json(json_path_person)
garment_pose = load_pose_from_json(json_path_garment)

pose_dim = person_pose.size(-1)  # Assuming person and garment have the same pose dimension
embedding_dim = 128  # You can adjust this value

person_pose_embedding_module = PoseEmbedding(pose_dim, embedding_dim)
garment_pose_embedding_module = PoseEmbedding(pose_dim, embedding_dim)

person_pose_embedding = person_pose_embedding_module(person_pose)
garment_pose_embedding = garment_pose_embedding_module(garment_pose)

class FiLM(nn.Module):
    def __init__(self, channels):
        super(FiLM, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return self.gamma * x + self.beta

class ResBlk(nn.Module):
    def __init__(self, channels):
        super(ResBlk, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.film = FiLM(channels)

    def forward(self, x):
        return x + self.film(self.conv(x))

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1)**0.5), dim=-1)
        out = torch.matmul(attention, V)
        return self.gamma * out + x

class CrossAttention(nn.Module):
    def __init__(self, channels):
        super(CrossAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        Q = self.query(x)
        K = self.key(y)
        V = self.value(y)

        attention = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1)**0.5), dim=-1)
        out = torch.matmul(attention, V)
        return self.gamma * out + x

class CLIP1DAttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super(CLIP1DAttentionPooling, self).__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(x * attention_weights, dim=1)
    
class PersonUNet(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super(PersonUNet, self).__init__()

        self.pose_embedding_module = PoseEmbedding(pose_dim, embedding_dim)
        self.clip_pooling = CLIP1DAttentionPooling(embedding_dim)

        # Initial 3x3 convolution before encoding
        self.init_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # Contracting Path with FiLM, ResBlk, SelfAttention, and CrossAttention
        self.enc1 = nn.Sequential(
            self.conv_block(in_channels, 64),
            FiLM(64),
            ResBlk(64),
            SelfAttention(64)
        )
        self.enc2 = nn.Sequential(
            self.conv_block(64, 128),
            *[ResBlk(128) for _ in range(3)],  # Resolution 128 repeated 3 times
            FiLM(128)
        )
        self.enc3 = nn.Sequential(
            self.conv_block(128, 256),
            *[ResBlk(256) for _ in range(2)],  # Resolution 256 repeated 2 times
            FiLM(256)
        )
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Cross Attention at resolution 16 repeated 7 times
        self.cross_attentions = nn.ModuleList([CrossAttention(64, 64) for _ in range(7)])

        # Expanding Path
        self.up3 = nn.Sequential(
            self.upconv_block(512, 256),
            *[ResBlk(256) for _ in range(2)],  # Resolution 64 repeated 4 times
            FiLM(256)
        )
        self.up2 = nn.Sequential(
            self.upconv_block(256, 128),
            *[ResBlk(128) for _ in range(3)],  # Resolution 32 repeated 7 times
            FiLM(128)
        )
        self.up1 = self.upconv_block(128, 64)

        # 3x3 convolution after decoding
        self.final_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, Ia, zt, person_pose, garment_pose, garment_features=None):
        # Concatenate Ia and zt along the channel dimension
        x = torch.cat([Ia, zt], dim=1)

        person_pose_embedding = self.pose_embedding_module(person_pose)
        garment_pose_embedding = self.pose_embedding_module(garment_pose)

        # Fusing pose embeddings using attention mechanism
        fused_pose_embedding = person_pose_embedding + garment_pose_embedding
        fused_pose_embedding = self.clip_pooling(fused_pose_embedding)

        x = self.init_conv(x)  # Initial convolution before encoding

        # Contracting Path
        enc1 = self.enc1(x)
        if garment_features:
            enc1 = self.cross_attentions[0](enc1, garment_features[0])
        enc2 = self.enc2(self.pool(enc1))
        if garment_features:
            enc2 = self.cross_attentions[1](enc2, garment_features[1])
        enc3 = self.enc3(self.pool(enc2))
        if garment_features:
            enc3 = self.cross_attentions[2](enc3, garment_features[2])
        enc4 = self.enc4(self.pool(enc3))
        if garment_features:
            enc4 = self.cross_attentions[3](enc4, garment_features[3])

        # Expanding Path with skip connections
        dec3 = self.up3(enc4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.final_conv(dec1)  # 3x3 convolution after decoding

        return self.out_conv(dec1)

class GarmentUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GarmentUNet, self).__init__()

        # Initial 3x3 convolution before encoding
        self.init_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # Contracting Path with FiLM and ResBlk (SelfAttention and CrossAttention removed)
        self.enc1 = nn.Sequential(
            self.conv_block(in_channels, 64),
            FiLM(64),
            ResBlk(64)
        )
        self.enc2 = nn.Sequential(
            self.conv_block(64, 128),
            *[ResBlk(128) for _ in range(3)],  # Resolution 128 repeated 3 times
            FiLM(128)
        )
        self.enc3 = nn.Sequential(
            self.conv_block(128, 256),
            *[ResBlk(256) for _ in range(2)],  # Resolution 256 repeated 2 times
            FiLM(256)
        )
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expanding Path (CrossAttention removed)
        self.up3 = nn.Sequential(
            self.upconv_block(512, 256),
            *[ResBlk(256) for _ in range(2)],  # Resolution 64 repeated 4 times
            FiLM(256)
        )
        self.up2 = nn.Sequential(
            self.upconv_block(256, 128),
            *[ResBlk(128) for _ in range(3)],  # Resolution 32 repeated 7 times
            FiLM(128)
        )
        self.up1 = self.upconv_block(128, 64)

        # 3x3 convolution after decoding
        self.final_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, Ic):
        x = self.init_conv(Ic)

        # Contracting Path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Extracting features at resolution 16 for the conditional pathway
        conditional_features = enc4

        # Expanding Path with skip connections
        dec3 = self.up3(enc4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)

        # Return all the feature maps for cross attention with PersonUNet (if needed in the future)
        return [enc1, enc2, enc3, enc4, dec2]
    
class TryOnDiffusionModel(nn.Module):
    def __init__(self):
        super(TryOnDiffusionModel, self).__init__()
        
        self.person_model = PersonUNet(in_channels=2, out_channels=1, embedding_dim=128)  # Ia와 zt 모두 1 채널이라고 가정
        self.garment_model = GarmentUNet(in_channels=1, out_channels=1)  # Ic가 1 채널이라고 가정
        self.super_resolution_unet = PersonUNet(in_channels=3, out_channels=1, embedding_dim=128)  # 256x256 Parallel-UNet

    def forward(self, Ia, zt, person_pose, garment_pose, Ic):
        garment_features = self.garment_model(Ic)
        I_128 = self.person_model(Ia, zt, person_pose, garment_pose, garment_features)
        
        I_128_upsampled = F.interpolate(I_128, scale_factor=2, mode='bilinear', align_corners=True)
        concatenated_input = torch.cat([I_128_upsampled, Ia, Ic], dim=1)  # Concatenate along the channel dimension
        I_256 = self.super_resolution_unet(concatenated_input)
        
        return I_256

# Example usage:
output = TryOnDiffusionModel()
I_256_result = output(Ia, zt, person_pose, garment_pose, Ic)

print(output)
