import pdb
import os
import ast
import shutil
import inspect
from pathlib import Path
from typing import Callable, Optional, Tuple, Union, Dict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def parse_config_file(cfg_file):
    with open(cfg_file, encoding='utf-8') as f:
        parsed_codes = ast.parse(f.read())
    codeobj = compile(parsed_codes, '', mode='exec')
    global_locals_var = {}
    eval(codeobj, global_locals_var)
    cfg_dict = {key: value
                for key, value in global_locals_var.items()
                if (not key.startswith('__'))
                }
    return cfg_dict

def save_config_file(cfg_file, save_dir):
    file_name = cfg_file.split('/')[-1]
    dest_file_name = os.path.join(save_dir, file_name)
    shutil.copy(cfg_file, dest_file_name)

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']
    

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    string = colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items())
    return string

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CombinedMeter:
    def __init__(self, infos: Dict = {}):
        self.meters = {}
        if not infos:
            self._is_initialized = False
        else:
            self.initialize(infos)

    def is_initialized(self,):
        return self._is_initialized 
            
    def initialize(self, infos):
        for key,value in infos.items():
            self.meters[key] = AverageMeter()
            self.meters[key].update(value)
        self._is_initialized = True
            
    def update(self, infos: Dict):
        if not self._is_initialized:
            self.initialize(infos)
            
        for key,value in infos.items():
            self.meters[key].update(value)

    def get_avg_info(self):
        results = {}
        for key, meter in self.meters.items():
            results[key] = meter.avg
        return results

    def reset(self):
        for key, meter in self.meters.items():
            meter.reset()
