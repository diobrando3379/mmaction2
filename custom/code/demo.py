import os
os.chdir('/home/diobrando/workspace/mmaction2')
print("Current working directory:", os.getcwd())
from mmengine.runner import load_checkpoint
from mmaction.apis import init_recognizer
from mmengine.model import BaseModel
import torch

cfg = 'configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py'
ckpt = '/home/diobrando/workspace/mmaction2/custom/pre_pth/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth'
model = init_recognizer(cfg, ckpt, device='cuda:0')
print(type(model.backbone))

x = torch.randn(2, 3, 32, 224, 224).cuda()
with torch.no_grad():
    y = model.backbone(x)
print(isinstance(y, tuple), [t.shape for t in y])