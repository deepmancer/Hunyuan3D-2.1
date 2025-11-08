import os
import sys
import warnings
import gc
import math
from pathlib import Path

try:
    import bpy  # noqa: F401  # Optional: available only inside Blender
except ImportError:
    bpy = None
import time

# --- Path setup ---
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, "modules/Hunyuan3D-2.1")
sys.path.insert(0, str(ROOT_DIR / 'hy3dpaint'))
sys.path.insert(0, str(ROOT_DIR / 'hy3dshape'))

# --- Third-party imports ---
import cv2
import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from trimesh import repair

# --- Local imports ---
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.preprocessors import ImageProcessorV2, array_to_tensor
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    import diso  # noqa: F401  # Optional dependency
    _HAS_DISO = True
except ImportError:
    _HAS_DISO = False

# torch.set_float32_matmul_precision('high')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")                                      
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")



import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ben2 import BEN_Base

torch.set_float32_matmul_precision(['high', 'highest'][0])

class BackgroundRemover:
    
    def __init__(self, device: str | torch.device = 'cuda'):
        self.device = torch.device(device)
        self.birefnet = self._init_birefnet()
    
    def _init_birefnet(self) -> nn.Module:
        model = BEN_Base.from_pretrained("PramaLLC/BEN2").to(self.device)
        model.eval()
        return model

    def remove_background(self, image: Image.Image, refine_foreground: bool = False) -> tuple[Image.Image, Image.Image]:
        foreground = self.birefnet.inference(image, refine_foreground=refine_foreground)
        alpha = foreground.getchannel('A')
        mask = ((np.array(alpha) / 255.0) > 0.85).astype(np.uint8) * 255
        return foreground, Image.fromarray(mask)


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

def post_process_mesh(mesh):
    """
    Post-process Hunyuan3D mesh output to ensure clean, sharp, watertight geometry with correct normals.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("post_process_mesh expects a trimesh.Trimesh instance")

    mesh = mesh.copy()
    print(f"  Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 1: Remove degenerate and duplicate faces
    nondegenerate_mask = mesh.nondegenerate_faces()
    if not np.all(nondegenerate_mask):
        mesh.update_faces(nondegenerate_mask)
        print(f"  Removed degenerate faces -> {len(mesh.faces)} faces")
    
    unique_faces = mesh.unique_faces()
    if len(unique_faces) != len(mesh.faces):
        mesh.update_faces(unique_faces)
        print(f"  Removed duplicate faces -> {len(mesh.faces)} faces")
    
    # Step 2: Keep only the largest connected component
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        print(f"  Found {len(components)} components, keeping largest")
        mesh = max(components, key=lambda m: m.area)
        print(f"  Largest component: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 3: Merge nearby vertices to clean up topology (preserves sharp edges)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    
    # Step 4: Repair holes to make watertight
    if not mesh.is_watertight:
        print("  Filling holes to make watertight...")
        mesh.fill_holes()
        if mesh.is_watertight:
            print("  Mesh is now watertight")
    
    # Step 5: Fix normal orientation for consistent winding
    mesh.fix_normals()
    
    # Step 6: Final cleanup
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    
    # Step 7: Ensure correct data types
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
    
    shape_mesh_dir = output_dir / 'shape_mesh'
    preprocessed_img_dir = output_dir / 'preprocessed_image'
    shape_mesh_dir.mkdir(exist_ok=True)
    preprocessed_img_dir.mkdir(exist_ok=True)
    
    mesh_path = shape_mesh_dir / f'{image_name}.glb'
    matted_image_path = Path(image_path)

    # Load shape pipeline
    print("Loading Hunyuan3D shape model...")
    model_path = 'tencent/Hunyuan3D-2.1'
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    image_processor = ImageProcessorV2(2048, border_ratio=0.0)
    pipeline.image_processor = image_processor
    
    with Image.open(matted_image_path) as image_pil:
        processed_result = image_processor(image_pil)
        processed_img = processed_result['image']

    processed_img_path = preprocessed_img_dir / f'{image_name}.png'
    processed_img_pil = torch_to_pil(processed_img)
    processed_img_pil.save(str(processed_img_path))
    
    # Setup generator
    # generator = torch.Generator(device=pipeline.device).manual_seed(seed)
   
    # Check for differentiable marching cubes support
    if _HAS_DISO:
        mc_algo = 'dmc'
        print("Using differentiable marching cubes (dmc)")
    else:
        mc_algo = 'mc'
        print("Warning: diso package not found. Using standard marching cubes.")
        print("Install with `pip install diso` to enable differentiable marching cubes.")
    
    # Run inference
    print("Running shape inference...")
    mesh = pipeline(
        image=str(matted_image_path),
        # generator=generator,
        output_type='trimesh',
        enable_pbar=True,
        guidance_scale=5.5,
        num_inference_steps=80,
        octree_resolution=512,
        # box_v=1.01,
        mc_algo=mc_algo,
        num_chunks=20000,
    )

    # Handle output
    if isinstance(mesh, (list, tuple)):
        mesh = mesh[0]
    
    if mesh is None:
        raise RuntimeError("Mesh generation failed; check the input images and GPU memory availability")
    
    # Validate mesh before post-processing
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
    
    if len(mesh.vertices) == 0:
        raise RuntimeError("Generated mesh has no vertices")
    
    if len(mesh.faces) == 0:
        raise RuntimeError("Generated mesh has no faces")
    
    print(f"Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Post-process mesh: make watertight and keep largest component
    print("Post-processing mesh...")
    mesh = post_process_mesh(mesh)
    
    # Save mesh with error handling
    print(f"Saving mesh to: {mesh_path}")
    try:
        # Validate mesh before export
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("Mesh has no vertices or faces after post-processing")
        
        # Attempt export
        mesh.export(mesh_path)
        print(f"  Successfully saved mesh to {mesh_path}")
    except Exception as e:
        print(f"  Error during mesh export: {e}")
        print(f"  Attempting fallback export to OBJ format...")
        try:
            # Fallback: try OBJ format which is more forgiving
            fallback_path = mesh_path.with_suffix('.obj')
            mesh.export(fallback_path)
            print(f"  Saved as OBJ format to: {fallback_path}")
            # Update mesh_path for return value
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
def run_texture_inference(image_path, mesh_path, output_dir, max_num_view=6, resolution=1024):
    output_dir = Path(output_dir)
    image_path = Path(image_path)
    mesh_path = Path(mesh_path)

    image_name = image_path.stem
    textured_mesh_dir = output_dir / 'textured_mesh' / image_name
    textured_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Initialize paint pipeline
    print("Loading Hunyuan3D paint pipeline...")
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.multiview_cfg_path = "/workspace/Hairdar/modules/Hunyuan3D-2.1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "/workspace/Hairdar/modules/Hunyuan3D-2.1/hy3dpaint/hunyuanpaintpbr"
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    
    textured_mesh_path = textured_mesh_dir / 'textured_mesh.glb'

    output_mesh_path = paint_pipeline(
        mesh_path=str(mesh_path),
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

def refine_alpha(rgba_u8: np.ndarray, radius: int = 8, eps: float = 1e-4, tiny_feather_sigma: float = 0.5) -> np.ndarray:
    # return rgba_u8[..., 3]  # Placeholder: skip refinement for now
    assert rgba_u8.dtype == np.uint8 and rgba_u8.shape[-1] == 4, "Expect uint8 RGBA"
    rgb_u8 = rgba_u8[..., :3]
    a_u8   = rgba_u8[..., 3]

    guide_gray_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGBA2GRAY) if rgb_u8.shape[-1] == 4 else \
                    cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)

    a_f = a_u8.astype(np.float32) / 255.0

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        a_ref = cv2.ximgproc.guidedFilter(guide_gray_u8, a_f, radius, eps)
    else:
        a8 = (a_f * 255).astype(np.uint8)
        a8_ref = cv2.bilateralFilter(a8, d=-1, sigmaColor=12, sigmaSpace=4)
        a_ref = a8_ref.astype(np.float32) / 255.0

    if tiny_feather_sigma and tiny_feather_sigma > 0:
        a_ref = cv2.GaussianBlur(a_ref, (0, 0), sigmaX=tiny_feather_sigma)

    a_ref_u8 = np.clip(a_ref * 255.0, 0, 255).astype(np.uint8)
    return a_ref_u8

def center_matted_image(image_path: Path, output_path: Path, bbox_ratio: float = 0.85):
    """Center foreground object in RGBA image while preserving transparency."""
    # Load image as RGBA
    img = Image.open(image_path).convert("RGBA")
    img_np = np.array(img, dtype=np.uint8)
    background_remover = BackgroundRemover()
    if img_np.shape[2] != 4:
        raise ValueError(f"Image must have an alpha channel (RGBA), got shape: {img_np.shape}")
    
    # Load silhouette mask from silhouette_mask folder
    image_stem = image_path.stem
    silhouette_mask_dir = image_path.parent.parent / 'silhouette_mask'
    silhouette_mask_path = silhouette_mask_dir / f'{image_stem}.png'
    
    if not silhouette_mask_path.exists():
        raise FileNotFoundError(f"Silhouette mask not found: {silhouette_mask_path}")
    
    # Load silhouette mask as grayscale
    silh_mask = Image.open(silhouette_mask_path).convert("L")
    silh_mask_np = np.array(silh_mask, dtype=np.uint8)
    
    # Verify mask dimensions match image dimensions
    if silh_mask_np.shape[:2] != img_np.shape[:2]:
        raise ValueError(f"Silhouette mask dimensions {silh_mask_np.shape[:2]} do not match image dimensions {img_np.shape[:2]}")
    
    # Split into RGB and use silhouette mask as alpha
    rgb = img_np[:, :, :3]
    alpha = silh_mask_np  # Use silhouette mask as foreground mask
    
    canvas_h, canvas_w = img_np.shape[:2]
    
    # STEP 1: First, align the foreground to the bottom of the canvas
    # Find bounding box of non-transparent pixels in original image
    coords = cv2.findNonZero(alpha)
    if coords is None:
        raise ValueError(f"No foreground region found in image: {image_path}")
    
    x_orig, y_orig, w_orig, h_orig = cv2.boundingRect(coords)
    
    # Find the bottom-most non-transparent row in the original image
    nonzero_rows = np.where(np.any(alpha > 0, axis=1))[0]
    if len(nonzero_rows) == 0:
        raise ValueError(f"No non-transparent pixels found in image")
    
    bottom_row_orig = nonzero_rows[-1]
    padding_below_orig = (canvas_h - 1) - bottom_row_orig
    
    # If there's padding at the bottom, shift the entire image up to remove it
    if padding_below_orig > 0:
        # Create new arrays for bottom-aligned image
        rgb_bottom_aligned = np.zeros_like(rgb)
        alpha_bottom_aligned = np.zeros_like(alpha)
        
        # Shift everything down by padding_below_orig
        rgb_bottom_aligned[padding_below_orig:, :] = rgb[:-padding_below_orig, :]
        alpha_bottom_aligned[padding_below_orig:, :] = alpha[:-padding_below_orig, :]
        
        # Update rgb and alpha to use bottom-aligned versions
        rgb = rgb_bottom_aligned
        alpha = alpha_bottom_aligned
    
    # STEP 2: Now work with the bottom-aligned image
    # Find bounding box of non-transparent pixels after bottom alignment
    coords = cv2.findNonZero(alpha)
    if coords is None:
        raise ValueError(f"No foreground region found after bottom alignment")
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Extract foreground region (RGB + alpha)
    fg_rgb = rgb[y:y+h, x:x+w]
    fg_alpha = alpha[y:y+h, x:x+w]
    
    canvas_h, canvas_w = img_np.shape[:2]
    
    # Calculate scale to fit within bbox_ratio of canvas
    max_size = int(min(canvas_h, canvas_w) * bbox_ratio)
    scale = min(max_size / w, max_size / h)
    
    # Resize if needed
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        fg_rgb = cv2.resize(fg_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fg_alpha = cv2.resize(fg_alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h
    
    # Create transparent canvas
    centered_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    centered_alpha = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    # STEP 3: Place the foreground centered horizontally and at the bottom vertically
    x_offset = (canvas_w - new_w) // 2  # Center horizontally (equal left/right padding)
    y_offset = canvas_h - new_h  # Align to bottom (foreground already bottom-aligned from Step 1)
    
    # Place foreground with horizontal centering and bottom alignment
    centered_rgb[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = fg_rgb
    centered_alpha[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = fg_alpha
    
    # Combine RGB and alpha
    centered_img = np.dstack([centered_rgb, centered_alpha])
    # Convert to RGB PIL Image (removing alpha channel)
    centered_img_pil = Image.fromarray(centered_img)
    centered_img_rgb = centered_img_pil.convert("RGB")
    centered_image_rgba, centered_image_mask = background_remover.remove_background(centered_img_rgb, refine_foreground=True)
    
    
    # # Ensure only the largest connected component is kept in the mask
    centered_mask_np = np.array(centered_image_mask, dtype=np.uint8)

    # # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(centered_mask_np, connectivity=8)

    if num_labels > 2:  # More than just background (label 0) and one foreground component
        # Find the largest component (excluding background at label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create new mask with only the largest component
        centered_mask_np = (labels == largest_label).astype(np.uint8) * 255
        
        print(f"  Removed {num_labels - 2} smaller connected components, keeping largest foreground region")

    # Apply the cleaned mask to the RGBA image
    centered_image_rgba_np = np.array(centered_image_rgba, dtype=np.uint8)
    centered_image_rgba_np[..., 3] = centered_mask_np 
    # Set background pixels (where mask is 0) to white
    centered_image_rgba_np[centered_mask_np == 0, :3] = 255
    centered_image_rgba = Image.fromarray(centered_image_rgba_np)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centered_image_rgba.save(output_path)
    print(f"Saved centered matted image to: {output_path}")

def run_single_inference(image_name: str, output_dir, force_recompute: bool = False):
    """Main function to run the shape inference pipeline."""
    fname = image_name
    matted_image_path = output_dir / 'matted_image_centered' / f'{fname}.png'

    if not matted_image_path.is_file():
        raise FileNotFoundError(f"Matted image not found: {matted_image_path}")

    mesh_path = output_dir / 'shape_mesh' / f'{fname}.glb'
    textured_mesh_path = output_dir / 'textured_mesh' / fname / 'textured_mesh.glb'

    if mesh_path.exists() and not force_recompute:
        print(f"Mesh already exists at {mesh_path}, skipping shape inference.")
    else:
        seed = int(time.time())
        run_shape_inference(
            image_path=str(matted_image_path),
            output_dir=str(output_dir),
            seed=seed
        )

    if textured_mesh_path.exists() and not force_recompute:
        print(f"Textured mesh already exists at {textured_mesh_path}, skipping texture inference.")
    else:
        textured_mesh_path = run_texture_inference(
            image_path=matted_image_path,
            mesh_path=mesh_path,
            output_dir=output_dir
        )

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

def main(data_dir: str = "/workspace/outputs"):
    output_dir = Path(data_dir)
    matted_images_dir = output_dir / 'image'
    matted_images_centered_dir = output_dir / 'matted_image_centered'
    matted_images_centered_dir.mkdir(parents=True, exist_ok=True)

    if not matted_images_dir.exists():
        raise FileNotFoundError(f"Matted images directory not found: {matted_images_dir}")

    # First pass: Center all matted images
    for image_file in matted_images_dir.glob("*.png"):
        image_name = image_file.stem
        centered_output_path = matted_images_centered_dir / f'{image_name}.png'
        render_image_output_path = output_dir / 'render' / f'{image_file.stem}.png'
        
        if render_image_output_path.exists():
            continue

        center_matted_image(
            image_path=image_file,
            output_path=centered_output_path,
            bbox_ratio=0.80,
        )
        
    import random
    # Second pass: Run inference on each image
    image_files = list(matted_images_centered_dir.glob("*.png"))
    random.shuffle(image_files)
    for image_file in image_files:
        render_image_output_path = output_dir / 'render' / f'{image_file.stem}.png'
        if render_image_output_path.exists():
            print(f"Rendered image already exists at {render_image_output_path}, skipping inference.")
            continue
        try:
            print(f"Processing image: {image_file}")
            run_single_inference(
                image_name=image_file.stem,
                output_dir=output_dir,
                force_recompute=True,
            )
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        
if __name__ == "__main__":
    main()