# 이 코드는 프로젝트에서 쓰이지 않았습니다.
# 추후 업데이트 할 예정입니다.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self):
        super(FiLM, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.gamma * x + self.beta

class CrossAttention(nn.Module):
    def __init__(self, in_channels=64, d=256):
        super(CrossAttention, self).__init__()
        self.scale = d ** 0.5
        self.query = nn.Conv2d(in_channels, d, 1)  # 입력 채널 수를 in_channels로 설정
        self.key = nn.Conv2d(in_channels, d, 1)    
        self.value = nn.Conv2d(in_channels, d, 1) 

    def forward(self, Jp_embedding, Jg_embedding):
        # 크기 확인 및 조정 코드 제거
        Q = self.query(Jp_embedding)
        K = self.key(Jg_embedding)
        V = self.value(Jg_embedding)
        
        batch_size, _, height, width = Q.size()
        Q = Q.view(batch_size, -1, height * width).permute(0, 2, 1)
        K = K.view(batch_size, -1, height * width)
        V = V.view(batch_size, -1, height * width).permute(0, 2, 1)  # V 텐서의 크기 조정

        attention = F.softmax(torch.matmul(Q, K) / self.scale, dim=-1)
        out = torch.matmul(attention, V).view(batch_size, -1, height, width)
        
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = (in_channels // 8) ** 0.5

    def forward(self, x):
        batch_size, C, width, height = x.size()

        Q = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # Q size: (B, W*H, C')
        K = self.key(x).view(batch_size, -1, width * height)  # K size: (B, C', W*H)
        V = self.value(x).view(batch_size, -1, width * height)  # V size: (B, C, W*H)

        attention = F.softmax(torch.matmul(Q, K) / self.scale, dim=-1)  # Attention size: (B, W*H, W*H)
        out = torch.matmul(attention, V.permute(0, 2, 1)).view(batch_size, C, width, height)

        return self.gamma * out + x
    
class UNetEncoder(nn.Module):
    def __init__(self, resolutions, repeats, channels, with_attention):
        super(UNetEncoder, self).__init__()
        layers = []
        in_ch = channels[0]  # 초기 입력 채널 수 설정

        for idx, (res, rep, ch, att) in enumerate(zip(resolutions, repeats, channels, with_attention)):
            for _ in range(rep):
                layers.append(FiLM())
                layers.append(ResidualBlock(in_ch, ch))  # 입력 채널 수와 출력 채널 수를 전달
                if att:
                    layers.append(CrossAttention(in_channels=64))
                    layers.append(SelfAttention(ch))
                in_ch = ch  # 다음 블록의 입력 채널 수를 현재 블록의 출력 채널 수로 설정
            if idx != len(resolutions) - 1:  # 마지막 resolution에서는 MaxPool을 추가하지 않음
                layers.append(nn.MaxPool2d(2, 2))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x, Jp_embedding=None, Jg_embedding=None):  # Jp_embedding 및 Jg_embedding 인수 추가
        outputs = []
        for layer in self.encoder:
            if isinstance(layer, CrossAttention):
                x = layer(Jp_embedding, Jg_embedding)
            else:
                x = layer(x)
            outputs.append(x)
        return outputs

class UNetDecoder(nn.Module):
    def __init__(self, resolutions, repeats, channels, with_attention, pose_embedding_dim):
        super(UNetDecoder, self).__init__()
        self.pose_embedding_dim = pose_embedding_dim
        self.layers = []
        self.pose_conv = nn.Conv2d(pose_embedding_dim, channels[0], kernel_size=3, padding=1)  # 크기 조정을 위한 합성곱 레이어
        
        in_ch = channels[0]  # 초기 입력 채널 수 설정

        for idx, (res, rep, ch, att) in enumerate(zip(resolutions, repeats, channels, with_attention)):
            for _ in range(rep):
                self.layers.append(FiLM())
                self.layers.append(ResidualBlock(in_ch, ch))  # 입력 채널 수와 출력 채널 수를 전달
                if att:
                    self.layers.append(CrossAttention(ch))
                    self.layers.append(SelfAttention(ch))
                in_ch = ch  # 다음 블록의 입력 채널 수를 현재 블록의 출력 채널 수로 설정
            self.layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.layers = self.layers[:-1]  # 마지막 Upsample 제거

    def forward(self, *encoded_layers, pose_embedding):
        x = encoded_layers[-1]  # 마지막 인코딩 레이어로 x 초기화
        pose_embedding = self.pose_conv(pose_embedding)  # 합성곱을 통해 pose_embedding 크기 조정
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, FiLM):
                x = layer(x)  # FiLM 레이어에 x만 전달
                x = x + pose_embedding  # 크기가 조정된 pose_embedding을 x에 더함
            elif isinstance(layer, nn.Upsample):
                # Upsampling 전에 skip-connection 추가
                x = torch.cat([x, encoded_layers[-(idx//2)-2]], dim=1)
                x = layer(x)
            else:
                x = layer(x)
        
        return x


class ParallelUNet128(nn.Module):
    def __init__(self):
        super(ParallelUNet128, self).__init__()
        
        resolutions = [128, 64, 32, 16]
        repeats = [3, 4, 6, 7]
        channels = [64, 128, 256, 512]
        with_attention = [False, False, True, True]
        
        # Pose Embeddings
        self.pose_embedding_Jp = nn.Linear(51, 64)
        self.pose_embedding_Jg = nn.Linear(51, 64)
        
        # HumanUNet
        self.human_initial_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.human_encoder = UNetEncoder(resolutions, repeats, channels, with_attention)
        self.human_decoder = UNetDecoder(resolutions[::-1], repeats[::-1], channels[::-1], with_attention[::-1], pose_embedding_dim=64)
        self.human_final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # GarmentUNet
        self.garment_initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.garment_encoder = UNetEncoder(resolutions, repeats, channels, with_attention)
        self.garment_decoder = UNetDecoder(resolutions[:2][::-1], repeats[:2][::-1], channels[:2][::-1], with_attention[:2][::-1], pose_embedding_dim=64)

    def forward(self, z_t, ctryon):
        Ia, Jp, Ic, Jg = ctryon
        
        # Pose Embeddings
        Jp_embedding = self.pose_embedding_Jp(Jp).unsqueeze(-1).unsqueeze(-1)
        Jg_embedding = self.pose_embedding_Jg(Jg).unsqueeze(-1).unsqueeze(-1)

        # Expand embeddings to desired height and width
        Jp_embedding = Jp_embedding.expand(Jp_embedding.size(0), -1, 1280, 720)
        Jg_embedding = Jg_embedding.expand(Jg_embedding.size(0), -1, 1280, 720)
        
        pose_info = Jp_embedding + Jg_embedding
        
        # GarmentUNet Encoding
        garment_input = self.garment_initial_conv(Ic)
        g_encoded_outputs = self.garment_encoder(garment_input, Jp_embedding, Jg_embedding)
        g_encoded_32, g_encoded_16 = g_encoded_outputs[-2], g_encoded_outputs[-1]
        
        # HumanUNet Encoding
        human_input = torch.cat([Ia, z_t], dim=1)
        human_input = self.human_initial_conv(human_input)
        h_encoded_outputs = self.human_encoder(human_input, Jp_embedding, Jg_embedding)
        h_encoded_128, h_encoded_64, h_encoded_32, h_encoded_16 = h_encoded_outputs
        
        # GarmentUNet Decoding
        g_decoded_32 = self.garment_decoder(g_encoded_16, g_encoded_32)
        
        # HumanUNet Decoding with GarmentUNet's output
        human_input_decoding = torch.cat([h_encoded_128, g_decoded_32], dim=1)
        h_decoded_128 = self.human_decoder(*h_encoded_outputs, pose_info)
        
        # Final output
        I128_tr = self.human_final_conv(h_decoded_128)
        
        return I128_tr

class ParallelUNet256(nn.Module):
    def __init__(self):
        super(ParallelUNet256, self).__init__()
        
        resolutions = [256, 128, 64, 32, 16]
        repeats = [2, 3, 4, 7, 7]
        channels = [128, 256, 512, 1024, 2048]
        with_attention = [False, False, False, True, True]
        
        # Pose Embeddings
        self.pose_embedding_Jp = nn.Linear(51, 64)
        self.pose_embedding_Jg = nn.Linear(51, 64)
        
        # HumanUNet
        self.human_initial_conv = nn.Conv2d(131, 128, kernel_size=3, padding=1)
        self.human_encoder = UNetEncoder(resolutions, repeats, channels, with_attention)
        self.human_decoder = UNetDecoder(resolutions[::-1], repeats[::-1], channels[::-1], with_attention[::-1], pose_embedding_dim=64)
        self.human_final_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        
        # GarmentUNet
        self.garment_initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.garment_encoder = UNetEncoder(resolutions[:-3], repeats[:-3], channels[:-3], with_attention[:-3])
        self.garment_decoder = UNetDecoder(resolutions[-3:-2][::-1], repeats[-3:-2][::-1], channels[-3:-2][::-1], with_attention[-3:-2][::-1], pose_embedding_dim=64)

    def forward(self, z_t, ctryon, I128_tr):
        Ia, Jp, Ic, Jg = ctryon
        I128_tr_upsampled = F.interpolate(I128_tr, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Pose Embeddings
        Jp_embedding = self.pose_embedding_Jp(Jp).unsqueeze(-1).unsqueeze(-1)
        Jg_embedding = self.pose_embedding_Jg(Jg).unsqueeze(-1).unsqueeze(-1)

        pose_info = Jp_embedding + Jg_embedding
        
        # GarmentUNet
        garment_input = self.garment_initial_conv(Ic)
        g_encoded_outputs = self.garment_encoder(garment_input, Jp_embedding, Jg_embedding)
        g_encoded_32, g_encoded_16 = g_encoded_outputs[-2], g_encoded_outputs[-1]
        g_decoded_16 = self.garment_decoder(g_encoded_32, g_encoded_16)  # Decode up to resolution 16
        
        # HumanUNet
        human_input = torch.cat([Ia, z_t, I128_tr_upsampled], dim=1)
        human_input = self.human_initial_conv(human_input)
        h_encoded_outputs = self.human_encoder(human_input, Jp_embedding, Jg_embedding)
        h_encoded_256, h_encoded_128, h_encoded_64, h_encoded_32, h_encoded_16 = h_encoded_outputs
        
        # Add skip connection from garmentUNet to personUNet at resolution 16
        h_encoded_16 += g_decoded_16
        
        # Continue with the rest of the HumanUNet decoding
        h_decoded_256 = self.human_decoder(*h_encoded_outputs, pose_info)
        
        human_output = self.human_final_conv(h_decoded_256)
        
        return human_output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.GroupNorm(min(1, in_channels), in_channels, affine=False),
            nn.SiLU(), 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            FiLM(),
            nn.GroupNorm(min(1, out_channels), out_channels, affine=False),
            nn.SiLU(),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        if isinstance(self.skip, nn.Identity):
            return x + self.main(x)
        else:
            return self.skip(x) + self.main(x)

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
 
        self.parallel_unet_128 = ParallelUNet128()
        self.parallel_unet_256 = ParallelUNet256()

    def forward(self, z_t, ctryon):
        
        I_128 = self.parallel_unet_128(z_t, ctryon)
        I_256 = self.parallel_unet_256(z_t, ctryon, I_128)  # I_128를 전달
        
        return I_256
    
import unittest
import torch

class TestModels(unittest.TestCase):

    def test_FiLM(self):
        film = FiLM()
        x = torch.randn(16, 64, 32, 32)
        out = film(x)
        self.assertEqual(x.shape, out.shape)
        self.assertTrue(torch.allclose(out, film.gamma * x + film.beta, atol=1e-7))

    def test_CrossAttention(self):
        cross_attention = CrossAttention()
        Jp_embedding = torch.randn((8, 64, 64, 64))
        Jg_embedding = torch.randn((8, 64, 64, 64))
        out = cross_attention(Jp_embedding, Jg_embedding)
        self.assertEqual(out.size(), (8, 256, 64, 64))  # 예상 출력 크기 수정

    def test_SelfAttention(self):
        self_attention = SelfAttention(in_channels=64)
        x = torch.randn((8, 64, 64, 64))
        out = self_attention(x)
        self.assertEqual(out.size(), (8, 64, 64, 64))


if __name__ == '__main__':
    unittest.main()
