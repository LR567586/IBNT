import torch
from thop import profile
from thop import clever_format
from models import MBR_model

model = MBR_model(575,  n_branches=[], n_groups=4, losses ="LBS", LAI=False, n_cams=0, n_views=0)
model.load_state_dict(torch.load(r'D:\MBR4B\logs\Veri776\MBR_4G\0\best_mAP.pt'))
model.eval()

input = torch.randn(1, 3, 256, 256)
cam = torch.tensor([0])
view = torch.tensor([0])

macs, params = profile(model, inputs=(input,cam,view))

macs, params = clever_format([macs, params], "%.3f")

print(f"FLOPs: {macs}, Parameters: {params}")
