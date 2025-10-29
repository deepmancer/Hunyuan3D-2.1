import os
import sys
import warnings
from pathlib import Path
import gc
try:
    import bpy  # noqa: F401  # Optional: available only inside Blender
except ImportError:
    bpy = None

import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from trimesh import repair

try:
    import diso  # noqa: F401  # Optional dependency
    _HAS_DISO = True
except ImportError:
    _HAS_DISO = False

from ben2 import BEN_Base

torch.set_float32_matmul_precision(['high', 'highest'][0])


try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


def enable_gpus():
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = list(cycles_preferences.devices)[:2]

    activated_gpus = []

    for device in devices:
        device.use = True
        activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = 'OPTIX'
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_persistent_data = True

    return activated_gpus

def torch_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    # Convert torch tensor to a PIL-friendly numpy array
    if isinstance(image_tensor, torch.Tensor):
        processed_img_np = image_tensor.detach().cpu().numpy()

        # Remove leading singleton batch dimensions
        while processed_img_np.ndim > 3 and processed_img_np.shape[0] == 1:
            processed_img_np = np.squeeze(processed_img_np, axis=0)

        if processed_img_np.ndim == 3:
            # Channel-first tensor (C, H, W)
            if processed_img_np.shape[0] in (1, 3, 4):
                processed_img_np = np.transpose(processed_img_np, (1, 2, 0))
            # If channel dimension is the last axis and is singleton, squeeze it
            if processed_img_np.shape[-1] == 1:
                processed_img_np = processed_img_np[..., 0]

        elif processed_img_np.ndim == 4:
            raise ValueError(
                f"Processed image has unexpected shape {processed_img_np.shape}; could not squeeze batch dimension"
            )

        # Normalize value range to [0, 255]
        processed_img_np = processed_img_np.astype(np.float32)
        img_min, img_max = processed_img_np.min(), processed_img_np.max()
        if img_max <= 1.0 and img_min >= 0.0:
            processed_img_np = processed_img_np * 255.0
        elif img_max <= 1.0 and img_min >= -1.0:
            processed_img_np = ((processed_img_np + 1.0) * 0.5) * 255.0

        processed_img_np = np.clip(processed_img_np, 0, 255).astype(np.uint8)

        processed_img_pil = Image.fromarray(processed_img_np)
    else:
        processed_img_pil = image_tensor
    return processed_img_pil

def render_mesh(fname, mesh_path, render_dir, resolution=1024):
    """Render a mesh (GLB/GLTF/OBJ/PLY) with Blender Cycles and save the image."""
    if bpy is None:
        raise RuntimeError("Blender Python API (bpy) is unavailable; run inside Blender 4.0+")

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Textured mesh not found: {mesh_path}")

    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    front_view_path = render_dir / f'{fname}.png'

    texture_root = mesh_path.parent
    diffuse_path = texture_root / 'textured_mesh.jpg'
    roughness_path = texture_root / 'textured_mesh_roughness.jpg'
    metallic_path = texture_root / 'textured_mesh_metallic.jpg'

    from mathutils import Matrix, Vector
    import addon_utils

    mesh_ext = mesh_path.suffix.lower()
    supported_ext = {'.glb', '.gltf', '.obj', '.ply'}
    if mesh_ext not in supported_ext:
        raise ValueError(f"Unsupported mesh format '{mesh_ext}'. Expected one of: {sorted(supported_ext)}")

    def _enable_addon(addon_name: str) -> None:
        try:
            addon_utils.enable(addon_name, default_set=True, persistent=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to enable Blender addon '{addon_name}': {exc}") from exc

    bpy.ops.wm.read_factory_settings(use_empty=True)

    if mesh_ext in {'.glb', '.gltf'}:
        _enable_addon('io_scene_gltf')
    elif mesh_ext == '.obj':
        _enable_addon('io_scene_obj')
    elif mesh_ext == '.ply':
        _enable_addon('io_mesh_ply')

    enable_gpus()

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 512
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.adaptive_min_samples = 64
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(front_view_path)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    view_layer = bpy.context.view_layer
    if hasattr(view_layer, "cycles"):
        view_layer.cycles.use_denoising = True
        if hasattr(view_layer.cycles, "denoiser"):
            has_optix = getattr(bpy.app.build_options, "optix", False)
            preferred = 'OPTIX' if has_optix else 'OPENIMAGEDENOISE'
            try:
                view_layer.cycles.denoiser = preferred
            except TypeError:
                # Fallback to Blender default if preferred denoiser unavailable
                view_layer.cycles.denoiser = 'NLM'

    # Disable world background for transparent rendering
    if scene.world:
        if scene.world.use_nodes:
            # Set background strength to 0 for transparency
            bg_node = scene.world.node_tree.nodes.get('Background')
            if bg_node is not None:
                bg_node.inputs['Strength'].default_value = 0.0
        else:
            # Create a minimal world setup with no background
            scene.world.use_nodes = True
            scene.world.node_tree.nodes.clear()
    else:
        # Create world if it doesn't exist
        scene.world = bpy.data.worlds.new("TransparentWorld")
        scene.world.use_nodes = True
        scene.world.node_tree.nodes.clear()

    existing_objects = set(bpy.data.objects)
    bpy.ops.object.select_all(action='DESELECT')

    if mesh_ext in {'.glb', '.gltf'}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif mesh_ext == '.obj':
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
    elif mesh_ext == '.ply':
        bpy.ops.wm.ply_import(filepath=str(mesh_path))
    imported_meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported_meshes:
        imported_meshes = [obj for obj in bpy.data.objects if obj not in existing_objects and obj.type == 'MESH']
    if not imported_meshes:
        raise RuntimeError(f"No mesh geometry imported from {mesh_path}")

    if mesh_ext in {'.glb', '.gltf'}:
        rotation_matrix = Matrix.Rotation(math.radians(90.0), 4, 'X')
        new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
        for obj in new_objects:
            obj.matrix_world = rotation_matrix @ obj.matrix_world
        bpy.context.view_layer.update()
        
    if mesh_ext in [".obj", ".ply"]:
        rotation_matrix = Matrix.Rotation(math.radians(180.0), 4, 'Y')
        new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
        for obj in new_objects:
            obj.matrix_world = rotation_matrix @ obj.matrix_world
        bpy.context.view_layer.update()
    
    material = bpy.data.materials.new(name='HairdarMaterial')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0.0, 0.0)
    material_output = nodes.new('ShaderNodeOutputMaterial')
    material_output.location = (200.0, 0.0)
    links.new(principled.outputs['BSDF'], material_output.inputs['Surface'])

    if diffuse_path.exists():
        color_tex = nodes.new('ShaderNodeTexImage')
        color_tex.image = bpy.data.images.load(str(diffuse_path), check_existing=True)
        color_tex.image.colorspace_settings.name = 'sRGB'
        color_tex.location = (-400.0, 0.0)
        links.new(color_tex.outputs['Color'], principled.inputs['Base Color'])
    else:
        principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)

    if roughness_path.exists():
        rough_tex = nodes.new('ShaderNodeTexImage')
        rough_tex.image = bpy.data.images.load(str(roughness_path), check_existing=True)
        rough_tex.image.colorspace_settings.name = 'Non-Color'
        rough_tex.location = (-400.0, -220.0)
        links.new(rough_tex.outputs['Color'], principled.inputs['Roughness'])
    else:
        principled.inputs['Roughness'].default_value = 0.45

    if metallic_path.exists():
        metal_tex = nodes.new('ShaderNodeTexImage')
        metal_tex.image = bpy.data.images.load(str(metallic_path), check_existing=True)
        metal_tex.image.colorspace_settings.name = 'Non-Color'
        metal_tex.location = (-400.0, -440.0)
        links.new(metal_tex.outputs['Color'], principled.inputs['Metallic'])
    else:
        principled.inputs['Metallic'].default_value = 0.0

    for obj in imported_meshes:
        if obj.data.materials:
            obj.data.materials.clear()
        obj.data.materials.append(material)

    bbox_min = Vector((float('inf'), float('inf'), float('inf')))
    bbox_max = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in imported_meshes:
        for coord in obj.bound_box:
            world_coord = obj.matrix_world @ Vector(coord)
            bbox_min.x = min(bbox_min.x, world_coord.x)
            bbox_min.y = min(bbox_min.y, world_coord.y)
            bbox_min.z = min(bbox_min.z, world_coord.z)
            bbox_max.x = max(bbox_max.x, world_coord.x)
            bbox_max.y = max(bbox_max.y, world_coord.y)
            bbox_max.z = max(bbox_max.z, world_coord.z)

    center = Vector((0.0, 0.0, 0.0))
    size = bbox_max - bbox_min
    max_dim = max(size.x, size.y, size.z, 1.0)
    cam_distance = 1.45

    camera_data = bpy.data.cameras.new(name='HairdarCamera')
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = 1.1
    camera_data.clip_start = 0.01
    camera_data.clip_end = cam_distance * 4.0

    camera_obj = bpy.data.objects.new('HairdarCamera', camera_data)
    camera_obj.location = center + Vector((0.0, -cam_distance, 0.0))
    camera_obj.rotation_euler = (np.radians(90.0), 0, 0)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj

    light_data = bpy.data.lights.new(name='HairdarKeyLight', type='AREA')
    light_data.energy = 1500.0
    light_data.size = 20
    light_obj = bpy.data.objects.new('HairdarKeyLight', light_data)
    light_obj.location = camera_obj.location
    light_obj.rotation_euler = camera_obj.rotation_euler
    scene.collection.objects.link(light_obj)
    # _orient_to_target(light_obj, center)

    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True, use_viewport=False)

    return front_view_path


def get_landmarks(image_path, lmk_output_dir, smplx_output_dir, smplx_transforms_output_dir):
    preprocess_modules_dir = "/localhome/aha220/Hairdar"
    print(f"Adding preprocess modules to sys.path: {preprocess_modules_dir}")
    sys.path.insert(0, str(preprocess_modules_dir))
    

    from data.preprocess.estimate_lmk import run_all as lmk_run_all
    from data.preprocess.estimate_smplx import run_all as smplx_run_all
    from data.preprocess.estimate_smplx_fixed_camera import run_all as smplx_fixed_camera_run_all

    lmk_run_all(
        input_dir=image_path,
        output_dir=lmk_output_dir,
    )
    
    # smplx_run_all(
    #     input_dir=image_path,
    #     lmk_dir=lmk_output_dir,
    #     output_dir=smplx_output_dir,
    # )
    
    # smplx_fixed_camera_run_all(
    #     input_dir=image_path,
    #     lmk_dir=lmk_output_dir,
    #     smplx_params_dir=smplx_output_dir,
    #     output_dir=smplx_transforms_output_dir,
    # )

def fit_flame(images_dir, output_dir):
    from modules.DECA.demos.demo_reconstruct import inference_deca
    inference_deca(
        images_dir=str(images_dir),
        output_dir=str(output_dir),
        device='cuda',
        saveObj=True,
        saveVis=True,
        saveKpt=True,
        saveImages=True,
    )

def run_single_inference(image_name: str, output_dir):
    """Main function to run the shape inference pipeline."""
    # Input paths - use the image from the images_dir based on image_name
    src_image_path = output_dir / 'image' / f"{image_name}.png"
    if not src_image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {src_image_path}")

    fname = image_name
    dest_image_path = src_image_path
    matted_images_dir = output_dir / 'matted_image'
    matted_images_dir.mkdir(parents=True, exist_ok=True)
    matted_image_path = matted_images_dir / f'{fname}.png'
    mesh_path = output_dir / 'shape_mesh' / f'{fname}.glb'
    textured_mesh_path = output_dir / 'textured_mesh' / fname / 'textured_mesh.glb'
    render_dir = output_dir / 'render'
    render_dir.mkdir(parents=True, exist_ok=True)

    if (render_dir / f'{fname}.png').exists():
        print(f"Rendered image already exists at {render_dir / f'{fname}.png'}, skipping rendering.")
    else:
        render_mesh(
            fname=fname,
            mesh_path=textured_mesh_path,
            render_dir=render_dir,
        )
    
    lmk_output_dir = output_dir / 'lmk'
    smplx_output_dir = output_dir / 'smplx_params'
    smplx_transforms_output_dir = output_dir / 'smplx_transforms'

    get_landmarks(
        image_path=render_dir,
        lmk_output_dir=lmk_output_dir,
        smplx_output_dir=smplx_output_dir,
        smplx_transforms_output_dir=smplx_transforms_output_dir,
    )
    flame_output_dir = output_dir / 'deca_output'
    flame_output_dir.mkdir(parents=True, exist_ok=True)
    fit_flame(
        images_dir=matted_images_dir,
        output_dir=flame_output_dir
    )
    

def main(data_dir: str = "/localhome/aha220/Hairdar/modules/Hunyuan3D-2.1/outputs/"):
    output_dir = Path(data_dir)
    
    # Initialize background remover once for all images
    print("Loading background remover...")    
    images_dir = output_dir / 'image'
    # Convert WEBP, JPEG, JPG images to PNG first
    for ext in ['*.webp', '*.jpeg', '*.jpg']:
        for image_file in images_dir.glob(ext):
            print(f"Converting {image_file.name} to PNG format...")
            with Image.open(image_file) as img:
                png_path = image_file.with_suffix('.png')
                img.convert('RGB').save(png_path, 'PNG')
                image_file.unlink()  # Remove original file
                print(f"Converted and saved as {png_path.name}")
    
    for image_file in images_dir.glob("*.png"):
        print(f"Processing image: {image_file}")
        image_fname = image_file.stem
        print(f"Image filename: {image_fname}")
        # if image_fname == "005":
        #     continue
        run_single_inference(
            image_name=image_fname,
            output_dir=output_dir
        )
        
if __name__ == "__main__":
    main()