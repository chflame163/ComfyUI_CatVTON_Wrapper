"""Image process functions for ComfyUI nodes
by chflame https://github.com/chflame163

@author: chflame
@title: CatVTON_Wrapper
@nickname: CatVTON_Wrapper
@description: CatVTON warpper for ComfyUI
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import math
import numpy as np
import torch
import scipy.ndimage
# from tqdm import tqdm
from PIL import Image, ImageFilter
from .catvton.pipeline import CatVTONPipeline
from torchvision.transforms.functional import to_pil_image, to_tensor
from diffusers.image_processor import VaeImageProcessor
import folder_paths

def log(message:str, message_type:str='info'):
    name = 'LayerStyle'

    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# 😺dzNodes: {name} -> {message}")

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def expand_mask(mask:torch.Tensor, grow:int, blur:int) -> torch.Tensor:
    # grow
    c = 0
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in growmask:
        output = m.numpy()
        for _ in range(abs(grow)):
            if grow < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    # blur
    for idx, tensor in enumerate(out):
        pil_image = tensor2pil(tensor.cpu().detach())
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur))
        out[idx] = pil2tensor(pil_image)
    ret_mask = torch.cat(out, dim=0)
    return ret_mask


def resize_and_padding_image(image:Image, size:tuple, background_color:str="#FFFFFF") -> tuple:
    # Padding to size ratio
    w, h = image.size
    target_w, target_h = size
    if w / h < target_w / target_h: # target更宽，补左右
        new_h = target_h
        new_w = w * target_h // h
    else:
        new_w = target_w
        new_h = h * target_w // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    # padding
    padding = Image.new("RGB", size, color=background_color)
    paste_coordinate = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    padding.paste(image, paste_coordinate)
    return padding, (paste_coordinate[0], paste_coordinate[1], paste_coordinate[0] + new_w, paste_coordinate[1] + new_h)

def restore_padding_image(image:Image, orig_size:tuple, bbox:tuple) -> Image:
    w, h = image.size
    orig_w, orig_h = orig_size
    ret_image = image.crop(bbox)
    return ret_image.resize((orig_w, orig_h), Image.LANCZOS)