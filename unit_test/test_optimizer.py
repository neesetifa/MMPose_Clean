# 工作路径切换到上一层
import os
import sys
ori_path = os.getcwd()
new_path = os.path.dirname(ori_path)
os.chdir(new_path)
sys.path.append(new_path)

import pdb
import torch
from torch.optim import *

from optim import build_scheduler, build_optimizer_wrapper

class Dummy(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.layer = torch.nn.Linear(10,10)

    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    # configs
    optim_wrapper = dict(optimizer=dict(type='Adam',
                                        lr=5e-4,
                                        )
                         )
    param_scheduler = [dict(type='LinearParamScheduler',
                            param_name='lr',
                            begin=0,
                            end=500,
                            start_factor=0.001,
                            by_epoch=False),  # warm-up
                       
                       dict(type='MultiStepParamScheduler',
                            param_name='lr',
                            begin=0,
                            end=210,
                            milestones=[170, 200],
                            gamma=0.1,
                            by_epoch=True)
                       ]

    # build code
    model = Dummy()
    optimizer_wrapper = build_optimizer_wrapper(optim_wrapper, model)
    schedulers = build_scheduler(param_scheduler, optimizer_wrapper)
    pdb.set_trace()
