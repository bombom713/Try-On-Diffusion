import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from tqdm import tqdm
from preprocessing_최적화 import get_pre
from model_최적화 import DiffusionModel

# 체크포인트 저장 함수
def save_checkpoint(epoch, model, optimizer, filename="checkpoint.pth.tar"):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)

# 체크포인트 불러오기 함수
def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> Loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch
    else:
        print(f"=> No checkpoint found at '{filename}'")
        return None

# Logging 설정
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 5
batch_size = 2
iterations = 100
initial_lr = 1e-6  # 초기 학습률을 작은 값으로 설정
final_lr = 1e-4
warmup_steps = 10000
conditioning_dropout_rate = 0.1

# Initialize model and optimizer
model = DiffusionModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=final_lr)

# Learning rate scheduler
def lr_schedule(step):
    if step < warmup_steps:
        return initial_lr + (final_lr - initial_lr) * (step / warmup_steps)
    return final_lr

# Define the loss function
def denoising_score_matching_loss(predicted, target, alpha_t, sigma_t):
    epsilon = torch.randn_like(target)
    z_t = alpha_t * target + sigma_t * epsilon
    return ((predicted - z_t) ** 2).mean()

# Training loop
ctryon, z_t, Ip = get_pre()
Ia, Jp, Ic, Jg = ctryon

# Save the original Ia for visualization later
Ia_original = Ia.clone()

# Move data to GPU
Ia = Ia.to(device)
Jp = Jp.to(device)
Ic = Ic.to(device)
Jg = Jg.to(device)
z_t = z_t.to(device)

# 학습 전에 체크포인트 불러오기 (필요한 경우)
start_epoch = load_checkpoint(model, optimizer)
if start_epoch is None:
    start_epoch = 0

try:
    for iteration in tqdm(range(start_epoch, iterations), desc="Training", ncols=100):
        # Apply conditioning dropout
        if torch.rand(1).item() < conditioning_dropout_rate:
            Ia = torch.zeros_like(Ia)
            Jp = torch.zeros_like(Jp)
            Ic = torch.zeros_like(Ic)
            Jg = torch.zeros_like(Jg)
        
        # Forward pass
        output = model(z_t, ctryon)
        
        # Compute loss
        alpha_t = 0.5
        sigma_t = 0.5
        loss = denoising_score_matching_loss(output, Ia, alpha_t, sigma_t)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule(iteration)

        # 일정한 간격으로 체크포인트 저장
        if iteration % 1000 == 0:
            save_checkpoint(iteration, model, optimizer)
            logging.info(f"Iteration [{iteration}/{iterations}], Loss: {loss.item():.4f}")
            print(f"Iteration [{iteration}/{iterations}], Loss: {loss.item():.4f}")

except Exception as e:
    logging.error(f"Error during training: {e}")
    raise e

# Visualization
import matplotlib.pyplot as plt

# 모델을 평가 모드로 전환
model.eval()

# 예측 이미지 생성
with torch.no_grad():
    generated_image = model(z_t, ctryon)

# 원본 이미지와 생성된 이미지 출력
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 원본 이미지 출력 (Dropout 적용 전의 원본 이미지 사용)
axes[0].imshow(Ia_original[0].permute(1, 2, 0).cpu().numpy())
axes[0].set_title("Original Image")
axes[0].axis("off")

# 생성된 이미지 출력
axes[1].imshow(generated_image[0].permute(1, 2, 0).cpu().numpy())
axes[1].set_title("Generated Image")
axes[1].axis("off")

plt.show()

