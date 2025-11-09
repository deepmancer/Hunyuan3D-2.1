import argparse
import gc
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh
from ben2 import BEN_Base
from PIL import Image

try:
    import bpy
except ImportError:
    bpy = None

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, "modules/Hunyuan3D-2.1")
sys.path.insert(0, str(ROOT_DIR / 'hy3dpaint'))
sys.path.insert(0, str(ROOT_DIR / 'hy3dshape'))

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.preprocessors import ImageProcessorV2
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    import diso
    _HAS_DISO = True
except ImportError:
    _HAS_DISO = False

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")


class BackgroundRemover:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.birefnet = self._init_birefnet()
    
    def _init_birefnet(self):
        model = BEN_Base.from_pretrained("PramaLLC/BEN2").to(self.device)
        model.eval()
        return model

    def remove_background(self, image, refine_foreground=False):
        foreground = self.birefnet.inference(image, refine_foreground=refine_foreground)
        alpha = foreground.getchannel('A')
        mask = ((np.array(alpha) / 255.0) > 0.85).astype(np.uint8) * 255
        return foreground, Image.fromarray(mask)


def enable_gpus(max_gpus=None):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    
    available_types = []
    try:
        for compute_type in ['OPTIX', 'HIP', 'CUDA', 'METAL', 'OPENCL']:
            try:
                cycles_preferences.compute_device_type = compute_type
                cycles_preferences.refresh_devices()
                gpu_devices = [d for d in cycles_preferences.devices if d.type != 'CPU']
                if gpu_devices:
                    available_types.append((compute_type, len(gpu_devices)))
                    print(f"Found {len(gpu_devices)} GPU(s) for {compute_type}")
            except (AttributeError, TypeError):
                continue
    except Exception as e:
        print(f"Warning during device detection: {e}")
    
    if not available_types:
        raise RuntimeError("No GPU compute devices available. Falling back to CPU is not implemented.")
    
    compute_type_priority = {
        'OPTIX': 3,
        'CUDA': 4,
        'HIP': 2,
        'METAL': 1,
        'OPENCL': 0
    }
    
    available_types.sort(key=lambda x: compute_type_priority.get(x[0], -1), reverse=True)
    selected_type = available_types[0][0]
    
    cycles_preferences.compute_device_type = selected_type
    cycles_preferences.refresh_devices()
    
    gpu_devices = [d for d in cycles_preferences.devices if d.type != 'CPU']
    if gpu_devices and selected_type == 'OPTIX':
        compute_gpu_indicators = ['H100', 'A100', 'A40', 'A30', 'A10', 'V100', 'P100', 'Tesla']
        for device in gpu_devices:
            if any(indicator in device.name for indicator in compute_gpu_indicators):
                if 'CUDA' in [t[0] for t in available_types]:
                    selected_type = 'CUDA'
                    print(f"Detected compute GPU ({device.name}), switching from OPTIX to CUDA for better performance")
                    break
    
    cycles_preferences.compute_device_type = selected_type
    cycles_preferences.refresh_devices()
    
    all_devices = cycles_preferences.devices
    gpu_devices = [d for d in all_devices if d.type != 'CPU']
    
    if not gpu_devices:
        raise RuntimeError(f"No GPU devices found for {selected_type}")
    
    devices_to_use = gpu_devices if max_gpus is None else gpu_devices[:max_gpus]
    
    activated_gpus = []
    for device in all_devices:
        if device in devices_to_use:
            device.use = True
            activated_gpus.append(device.name)
            print(f"Activated GPU: {device.name}")
        else:
            device.use = False
    
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_persistent_data = True
    
    return {'gpus': activated_gpus, 'compute_type': selected_type}

def torch_to_pil(image_tensor):
    if not isinstance(image_tensor, torch.Tensor):
        return image_tensor
    
    processed_img_np = image_tensor.detach().cpu().numpy()

    while processed_img_np.ndim > 3 and processed_img_np.shape[0] == 1:
        processed_img_np = np.squeeze(processed_img_np, axis=0)

    if processed_img_np.ndim == 3:
        if processed_img_np.shape[0] in (1, 3, 4):
            processed_img_np = np.transpose(processed_img_np, (1, 2, 0))
        if processed_img_np.shape[-1] == 1:
            processed_img_np = processed_img_np[..., 0]
    elif processed_img_np.ndim == 4:
        raise ValueError(
            f"Processed image has unexpected shape {processed_img_np.shape}; could not squeeze batch dimension"
        )

    processed_img_np = processed_img_np.astype(np.float32)
    img_min, img_max = processed_img_np.min(), processed_img_np.max()
    if img_max <= 1.0 and img_min >= 0.0:
        processed_img_np = processed_img_np * 255.0
    elif img_max <= 1.0 and img_min >= -1.0:
        processed_img_np = ((processed_img_np + 1.0) * 0.5) * 255.0

    processed_img_np = np.clip(processed_img_np, 0, 255).astype(np.uint8)

    return Image.fromarray(processed_img_np)

def post_process_mesh(mesh):
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("post_process_mesh expects a trimesh.Trimesh instance")

    mesh = mesh.copy()
    print(f"  Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    nondegenerate_mask = mesh.nondegenerate_faces()
    if not np.all(nondegenerate_mask):
        mesh.update_faces(nondegenerate_mask)
        print(f"  Removed degenerate faces -> {len(mesh.faces)} faces")
    
    unique_faces = mesh.unique_faces()
    if len(unique_faces) != len(mesh.faces):
        mesh.update_faces(unique_faces)
        print(f"  Removed duplicate faces -> {len(mesh.faces)} faces")
    
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        print(f"  Found {len(components)} components, keeping largest")
        mesh = max(components, key=lambda m: m.area)
        print(f"  Largest component: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    
    if not mesh.is_watertight:
        print("  Filling holes to make watertight...")
        mesh.fill_holes()
        if mesh.is_watertight:
            print("  Mesh is now watertight")
    
    mesh.fix_normals()
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    
    mesh.faces = mesh.faces.astype(np.int32)
    mesh.vertices = mesh.vertices.astype(np.float32)
    
    print(f"  Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Watertight: {mesh.is_watertight}, Winding consistent: {mesh.is_winding_consistent}")
    
    return mesh

@torch.inference_mode()
def run_shape_inference(image_path, output_dir, seed=1234):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shape_mesh_dir = output_dir / "hunyuannn" / image_name
    shape_mesh_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_img_dir = output_dir / 'preprocessed_image'
    shape_mesh_dir.mkdir(exist_ok=True)
    preprocessed_img_dir.mkdir(exist_ok=True)
    
    mesh_path = shape_mesh_dir / 'shape_mesh.glb'
    matted_image_path = Path(image_path)

    print("Loading Hunyuan3D shape model...")
    model_path = 'tencent/Hunyuan3D-2.1'
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    image_processor = ImageProcessorV2(1024, border_ratio=0.0)
    pipeline.image_processor = image_processor
    
    with Image.open(matted_image_path) as image_pil:
        processed_result = image_processor(image_pil)
        processed_img = processed_result['image']

    processed_img_path = preprocessed_img_dir / f'{image_name}.png'
    processed_img_pil = torch_to_pil(processed_img)
    processed_img_pil.save(str(processed_img_path))
    
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
   
    if _HAS_DISO:
        mc_algo = 'dmc'
        print("Using differentiable marching cubes (dmc)")
    else:
        mc_algo = 'mc'
        print("Warning: diso package not found. Using standard marching cubes.")
        print("Install with `pip install diso` to enable differentiable marching cubes.")
    
    print("Running shape inference...")
    mesh = pipeline(
        image=str(matted_image_path),
        generator=generator,
        output_type='trimesh',
        enable_pbar=True,
        guidance_scale=5.5,
        num_inference_steps=50,
        octree_resolution=384,
        mc_algo=mc_algo,
        num_chunks=20000,
    )

    if isinstance(mesh, (list, tuple)):
        mesh = mesh[0]
    
    if mesh is None:
        raise RuntimeError("Mesh generation failed; check the input images and GPU memory availability")
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    
    if len(mesh.vertices) == 0:
        raise RuntimeError("Generated mesh has no vertices")
    
    if len(mesh.faces) == 0:
        raise RuntimeError("Generated mesh has no faces")
    
    print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print("Post-processing mesh...")
    mesh = post_process_mesh(mesh)
    
    print(f"Saving mesh to: {mesh_path}")
    try:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Mesh has no vertices or faces after post-processing")
        
        mesh.export(mesh_path)
        print(f"  Successfully saved mesh to {mesh_path}")
    except Exception as e:
        print(f"  Error during mesh export: {e}")
        print(f"  Attempting fallback export to OBJ format...")
        try:
            fallback_path = mesh_path.with_suffix('.obj')
            mesh.export(fallback_path)
            print(f"  Saved as OBJ format to: {fallback_path}")
            mesh_path = fallback_path
        except Exception as fallback_error:
            print(f"  Fallback export also failed: {fallback_error}")
            raise RuntimeError(f"Failed to export mesh in both GLB and OBJ formats: {e}")

    torch.cuda.empty_cache()
    gc.collect()
    
    del pipeline

    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'mesh_path': mesh_path,
        'image_path': image_path,
        'matted_image_path': matted_image_path,
    }

@torch.inference_mode()
def run_texture_inference(image_path, shape_mesh_path, output_dir, max_num_view=4, resolution=512):
    output_dir = Path(output_dir)
    image_path = Path(image_path)
    shape_mesh_path = Path(shape_mesh_path)

    image_name = image_path.stem
    textured_mesh_dir = output_dir / "hunyuannn" / image_name
    textured_mesh_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Hunyuan3D paint pipeline...")
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.multiview_cfg_path = "/workspace/Hairdar/modules/Hunyuan3D-2.1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "/workspace/Hairdar/modules/Hunyuan3D-2.1/hy3dpaint/hunyuanpaintpbr"
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    
    textured_mesh_path = textured_mesh_dir / 'textured_mesh.glb'

    output_mesh_path = paint_pipeline(
        mesh_path=str(shape_mesh_path),
        image_path=str(image_path),
        output_mesh_path=str(textured_mesh_path)
    )

    torch.cuda.empty_cache()
    gc.collect()
    
    del paint_pipeline

    torch.cuda.empty_cache()
    gc.collect()

    return Path(output_mesh_path)

def render_mesh(fname, mesh_path, render_dir, resolution=1024):
    if bpy is None:
        raise RuntimeError("Blender Python API (bpy) is unavailable; run inside Blender 4.0+")

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Textured mesh not found: {mesh_path}")

    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    front_view_path = render_dir / f'{fname}.png'

    texture_root = mesh_path.parent
    materials_dir = texture_root / 'materials'
    diffuse_path = materials_dir / 'albedo.jpg'
    roughness_path = materials_dir / 'roughness.jpg'
    metallic_path = materials_dir / 'metallic.jpg'

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
                view_layer.cycles.denoiser = 'NLM'

    if scene.world:
        if scene.world.use_nodes:
            bg_node = scene.world.node_tree.nodes.get('Background')
            if bg_node is not None:
                bg_node.inputs['Strength'].default_value = 0.0
        else:
            scene.world.use_nodes = True
            scene.world.node_tree.nodes.clear()
    else:
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

    new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]
    
    if mesh_ext in {'.glb', '.gltf'}:
        rotation_matrix = Matrix.Rotation(math.radians(90.0), 4, 'X')
        for obj in new_objects:
            obj.matrix_world = rotation_matrix @ obj.matrix_world
        bpy.context.view_layer.update()
        
    if mesh_ext in {'.obj', '.ply'}:
        rotation_matrix = Matrix.Rotation(math.radians(180.0), 4, 'Y')
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

    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True, use_viewport=False)

    return front_view_path

def run_single_inference(image_name, output_dir, force_recompute=False):
    fname = image_name
    matted_image_path = output_dir / 'matted_image_centered' / f'{fname}.png'

    if not matted_image_path.is_file():
        raise FileNotFoundError(f"Matted image not found: {matted_image_path}")

    hunyuan_3d_output_dir = output_dir / 'hunyuannn'
    hunyuan_3d_output_dir.mkdir(parents=True, exist_ok=True)
    shape_mesh_path = hunyuan_3d_output_dir / fname / 'shape_mesh.glb'
    textured_mesh_path = hunyuan_3d_output_dir / fname / 'textured_mesh.glb'

    shape_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    textured_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    
    if textured_mesh_path.exists() and not force_recompute:
        print(f"Textured mesh already exists at {textured_mesh_path}, skipping texture inference.")
    else:
        if shape_mesh_path.exists() and not force_recompute:
            print(f"Mesh already exists at {shape_mesh_path}, skipping shape inference.")
        else:
            seed = int(time.time())
            run_shape_inference(
                image_path=str(matted_image_path),
                output_dir=str(output_dir),
                seed=seed
            )

        textured_mesh_path = run_texture_inference(
            image_path=matted_image_path,
            shape_mesh_path=shape_mesh_path,
            output_dir=output_dir
        )
        
        # Delete intermediate files
        files_to_delete = [
            shape_mesh_path.parent / 'shape_mesh.glb',
            textured_mesh_path.parent / 'textured_mesh.ply',
            textured_mesh_path.parent / 'white_mesh_remesh.obj'
        ]
        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                    print(f"Deleted intermediate file: {file_path}")
                except Exception as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
        
        # Rename and move material files
        materials_dir = textured_mesh_path.parent / 'materials'
        materials_dir.mkdir(parents=True, exist_ok=True)
        
        material_files = [
            ('textured_mesh_metallic.jpg', 'metallic.jpg'),
            ('textured_mesh_roughness.jpg', 'roughness.jpg'),
            ('textured_mesh.jpg', 'albedo.jpg')
        ]
        
        for old_name, new_name in material_files:
            src_path = textured_mesh_path.parent / old_name
            dst_path = materials_dir / new_name
            try:
                # First rename the file
                temp_path = textured_mesh_path.parent / new_name
                if src_path.exists():
                    src_path.rename(temp_path)
                    print(f"Renamed: {old_name} -> {new_name}")
                
                # Then move to materials directory
                if temp_path.exists():
                    temp_path.rename(dst_path)
                print(f"Moved: {new_name} to materials/")
            except Exception as e:
                print(f"Warning: Could not rename/move {old_name}: {e}")

    render_dir = output_dir / 'render'
    render_dir.mkdir(parents=True, exist_ok=True)

    if (render_dir / f'{fname}.png').exists() and not force_recompute:
        print(f"Rendered image already exists at {render_dir / f'{fname}.png'}, skipping rendering.")
    else:
        render_mesh(
            fname=fname,
            mesh_path=textured_mesh_path,
            render_dir=render_dir,
        )

def main(data_dir="/workspace/outputs", force_recompute=False):
    output_dir = Path(data_dir)
    matted_images_dir = output_dir / 'image'
    matted_images_centered_dir = output_dir / 'matted_image_centered'
    matted_images_centered_dir.mkdir(parents=True, exist_ok=True)

    if not matted_images_dir.exists():
        raise FileNotFoundError(f"Matted images directory not found: {matted_images_dir}")

    image_files = list(matted_images_centered_dir.glob("*.png"))
    random.shuffle(image_files)
    for image_file in image_files:
        try:
            print(f"Processing image: {image_file}")
            run_single_inference(
                image_name=image_file.stem,
                output_dir=output_dir,
                force_recompute=force_recompute,
            )
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunyuan3D Inference Pipeline")
    parser.add_argument('--data_dir', type=str, default="/workspace/outputs/",
                        help='Path to the data directory')
    parser.add_argument("--force_recompute", default=False,
                        help='Force recomputation of shape and texture inference')
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        force_recompute=args.force_recompute
    )