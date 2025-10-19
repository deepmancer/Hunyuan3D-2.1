import sys
import os
from pathlib import Path

try:
    import bpy  # noqa: F401  # Optional: available only inside Blender
except ImportError:
    bpy = None

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import cv2
import numpy as np
import torch
from PIL import Image

from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.preprocessors import ImageProcessorV2, array_to_tensor


from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

def ensure_rgba(path, remover):
    image = Image.open(path).convert("RGBA")
    alpha = np.array(image)[..., 3]
    if np.all(alpha == 255):
        print("Image has no transparency channel.")
        # No transparency: run optional background removal for better masks
        try:
            image = remover(image)
        except Exception:
            pass
    return np.array(image)


def compute_union_mask(rgba_images):
    shapes = [img.shape[:2] for img in rgba_images]
    if len({s for s in shapes}) != 1:
        raise ValueError("Input images must share the same resolution before preprocessing")
    masks = [img[..., 3] for img in rgba_images]
    union = np.maximum.reduce(masks)
    if union.max() == 0:
        raise ValueError("Empty alpha masks detected; please provide RGBA images or enable background removal")
    return union.astype(np.uint8)


def preprocess_with_shared_mask(processor, rgba_image, shared_mask, border_ratio=0.15):
    height, width = rgba_image.shape[:2]
    size = max(height, width)

    canvas_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((size, size), dtype=np.uint8)
    canvas_union = np.zeros((size, size), dtype=np.uint8)

    canvas_rgb[:height, :width] = rgba_image[..., :3]
    canvas_alpha[:height, :width] = rgba_image[..., 3]
    canvas_union[:height, :width] = shared_mask

    needs_fill = (canvas_union > 0) & (canvas_alpha == 0)
    canvas_rgb[needs_fill] = 255

    coords = np.nonzero(canvas_union)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2

    crop_rgb = canvas_rgb[x_min:x_max, y_min:y_max]
    crop_alpha = canvas_alpha[x_min:x_max, y_min:y_max]
    if crop_rgb.size == 0:
        raise ValueError("Crop contains no foreground pixels; verify the shared mask")
    crop_rgb_resized = cv2.resize(crop_rgb, (w2, h2), interpolation=cv2.INTER_AREA)
    crop_alpha_resized = cv2.resize(crop_alpha, (w2, h2), interpolation=cv2.INTER_NEAREST)

    result_rgb = np.ones((size, size, 3), dtype=np.uint8) * 255
    result_alpha = np.zeros((size, size), dtype=np.uint8)
    result_rgb[x2_min:x2_max, y2_min:y2_max] = crop_rgb_resized
    result_alpha[x2_min:x2_max, y2_min:y2_max] = crop_alpha_resized

    alpha_norm = (result_alpha.astype(np.float32) / 255.0)[..., None]
    composited = result_rgb.astype(np.float32) * alpha_norm + 255.0 * (1.0 - alpha_norm)
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    image_resized = cv2.resize(composited, (processor.size, processor.size), interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(result_alpha, (processor.size, processor.size), interpolation=cv2.INTER_NEAREST)[..., np.newaxis]

    image_tensor = array_to_tensor(image_resized)
    mask_tensor = array_to_tensor(mask_resized)

    metadata = {
        'border_ratio': border_ratio,
        'scale': scale,
        'bbox': [int(x_min), int(x_max), int(y_min), int(y_max)],
        'canvas_size': size,
    }
    return image_tensor, mask_tensor, metadata, image_resized.astype(np.uint8), mask_resized.astype(np.uint8)


def run_shape_inference(image_path, bald_image_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    hair_mesh_path = output_dir / 'demo.glb'
    bald_mesh_path = output_dir / 'demo_bald.glb'

    remover = BackgroundRemover()
    image_rgba = ensure_rgba(image_path, remover)
    bald_rgba = ensure_rgba(bald_image_path, remover)

    shared_mask = compute_union_mask([image_rgba, bald_rgba])
    processor = ImageProcessorV2(size=768, border_ratio=0.0)

    image_tensor, mask_tensor, meta_image, image_np, mask_np = preprocess_with_shared_mask(
        processor, image_rgba, shared_mask
    )
    bald_tensor, bald_mask_tensor, meta_bald, bald_np, bald_mask_np = preprocess_with_shared_mask(
        processor, bald_rgba, shared_mask
    )

    batch_image = torch.cat([image_tensor, bald_tensor], dim=0)
    batch_mask = torch.cat([mask_tensor, bald_mask_tensor], dim=0)

    Image.fromarray(image_np).save(output_dir / 'preprocessed_demo.png')
    Image.fromarray(mask_np.squeeze(-1)).save(output_dir / 'preprocessed_demo_mask.png')
    Image.fromarray(bald_np).save(output_dir / 'preprocessed_demo_bald.png')
    Image.fromarray(bald_mask_np.squeeze(-1)).save(output_dir / 'preprocessed_demo_bald_mask.png')
    Image.fromarray(shared_mask).save(output_dir / 'shared_alpha_mask.png')

    model_path = 'tencent/Hunyuan3D-2.1'
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    # pipeline.enable_flashvdm()
    generator = torch.Generator(device=pipeline.device).manual_seed(1234)
    batch_image = batch_image.to(device=pipeline.device, dtype=pipeline.dtype)
    batch_mask = batch_mask.to(device=pipeline.device, dtype=pipeline.dtype)

    try:
        import diso  # noqa: F401  # Verify that differentiable marching cubes is available
        # pipeline.vae.enable_flashvdm_decoder(mc_algo='dmc')
        # dmc_kwargs['mc_algo'] = 'dmc'
    except ImportError:
        print("Warning: diso package not found. Install with `pip install diso` to enable differentiable marching cubes.")
    meshes = pipeline(
        image=batch_image,
        mask=batch_mask,
        generator=generator,
        output_type='trimesh',
        enable_pbar=True,
        num_inference_steps=100,
        octree_resolution=768,
        mc_algo="dmc",
        guidance_scale=3.0,
        num_chunks=8000,
    )

    if not isinstance(meshes, (list, tuple)) or len(meshes) < 2:
        raise RuntimeError("Unexpected pipeline output; expected two meshes for the input batch")
    if meshes[0] is None or meshes[1] is None:
        raise RuntimeError("Mesh decoding failed; check the input images and GPU memory availability")

    

    meshes[0].export(hair_mesh_path)
    meshes[1].export(bald_mesh_path)

    meta_path = output_dir / 'preprocess_metadata.txt'
    with meta_path.open('w', encoding='utf-8') as f:
        f.write(f"image_meta: {meta_image}\n")
        f.write(f"bald_meta: {meta_bald}\n")

    return str(hair_mesh_path), str(bald_mesh_path)


# shape
image_path = "assets/demo.png"
bald_image_path = "assets/demo_bald.png"

hair_mesh_path, bald_mesh_path = run_shape_inference(
    image_path,
    bald_image_path,
    output_dir=Path('demo_output')
)

# # paint
# max_num_view = 6  # can be 6 to 9
# resolution = 512 # can be 768 or 512
# conf = Hunyuan3DPaintConfig(max_num_view, resolution)
# conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
# conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
# conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
# paint_pipeline = Hunyuan3DPaintPipeline(conf)

# output_mesh_path = 'demo_textured.glb'
# output_mesh_path = paint_pipeline(
#     mesh_path = hair_mesh_path,
#     image_path = 'assets/demo.png',
#     output_mesh_path = output_mesh_path
# )

# image_path = 
# mesh_path = "CB_LF1576A_2_mesh.glb"

# # let's generate a mesh first
# shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
# mesh_untextured = shape_pipeline(image='assets/demo.png')[0]

# paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=1024))
# mesh_textured = paint_pipeline(mesh_path, image_path='assets/demo.png')