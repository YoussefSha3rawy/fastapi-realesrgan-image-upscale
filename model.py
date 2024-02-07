import os
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import torch

from realesrgan import RealESRGANer

MODEL_NAME = 'RealESRGAN_x4plus'


def upscale_image(image: np.ndarray, scale: int = 4, face_enhance: bool = False):
    """
    upscale image
    """
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = [
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

    model_path = os.path.join('weights', MODEL_NAME + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        device=device,
        half=False)

    if face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=device)
        
    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(
                image, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(image, outscale=scale)

    except RuntimeError as error:
        print('Error', error)

    else:
        return output
