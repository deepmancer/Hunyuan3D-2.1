# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import numpy as np
from PIL import Image
import sys

class imageSuperNet:
    def __init__(self, config) -> None:
        sys.path.insert(0, "../../")
        from data.generative.super_res import CodeFormerEnhancer
        # from realesrgan import RealESRGANer
        # from basicsr.archs.rrdbnet_arch import RRDBNet

        # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        # upsampler = RealESRGANer(
        #     scale=4,
        #     model_path=config.realesrgan_ckpt_path,
        #     dni_weight=None,
        #     model=model,
        #     tile=0,
        #     tile_pad=10,
        #     pre_pad=0,
        #     half=True,
        #     gpu_id=None,
        # )
        self.upsampler = CodeFormerEnhancer(
            device="cuda",
            ultrasharp=True,
        )

        sys.path.pop(0)

    def __call__(self, image):
        output = self.upsampler.enhance(
            image,
            face_align=True,
            background_enhance=False,
            face_upsample=False,
            upscale=2,
            codeformer_fidelity=0.75,
        )
        if not isinstance(output, Image.Image):    
            output = Image.fromarray(output)
        return output
    