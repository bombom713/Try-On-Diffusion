# TryOnDiffusion<br>
▶TryOnDiffusion_A Tale of Two UNets 논문 탐색 후 Preprocessing과 Parelle UNet을 구현해 봤습니다. (스스로 학습용)<br>
<br>
※ 프로젝트에서는 진희님과 윤주님이 구현하신 Preprocessing에 준혁님이 구현하신 ParallelUNet에 작성자의 EfficientUNet을 결합한 형태로 작성했습니다.<br>
결합한 train코드는 지호님이 작성하셨습니다.<br>
진희님 github: "https://github.com/wlsl6569"
윤주님 github: "https://github.com/deeplearningb"
준혁님 github: "https://github.com/Mutoy-choi/Tryondiffusion"<br>
지호님 github: "https://github.com/capybarajh"
<br>
<br>
<br>
▶ Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding 논문 탐색 후 Efficient UNet을 구현해 봤습니다.<br>
구현해 낸 Efficient UNet은 TryonDiffusion_A Tale of Two UNets에 맞게 하이퍼파라미터를 조정했습니다.<br>
<br>
※ 작성자의 코드의 주요 사항:<br>
1. efficientnet을 encoder로 사용하기 때문에 up.conv 레이어를 사용하지 않고 upsampling을 통해 해상도를 상승시킵니다.<br>
2. 노이즈 조건화 증강(augmentation)은 준혁님이 구현하신 코드의 output을 입력데이터로 받아 적용시킵니다.<br>
3. attention layer의 bool값을 False로 지정해 완전 합성곱 구조를 구현합니다.<br>
(Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding 논문에서는 attention layer를 conditional하게 적용한다고 하였습니다. 따라서 논문 구현 코드의 의의를 두기 위해 제거하는 것이 아닌 False로 지정해 완전 합성곱 구조를 구현합니다.)
