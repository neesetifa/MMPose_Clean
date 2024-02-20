import torch
from collections import OrderedDict

WEIGHT = '202402181150_hm_mb_050/best.pth'

ww = torch.load(WEIGHT, map_location='cpu')['model']
new_ww = OrderedDict()
for k, v in ww.items():
    if 'backbone' in k:
        new_ww[k] = v

torch.save(new_ww, 'backbone.pth')
