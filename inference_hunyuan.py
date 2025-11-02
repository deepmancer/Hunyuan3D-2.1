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
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("post_process_mesh expects a trimesh.Trimesh instance")

    # Work on a copy to avoid mutating pipeline internals unexpectedly.
    try:
        mesh = mesh.copy()
    except Exception as e:
        print(f"  Warning: Failed to copy mesh: {e}")
        print(f"  Working with original mesh (may have side effects)")

    print(f"  Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Initial validation: check for invalid face indices
    num_vertices = len(mesh.vertices)
    if len(mesh.faces) > 0:
        max_face_idx = mesh.faces.max()
        min_face_idx = mesh.faces.min()
        if max_face_idx >= num_vertices or min_face_idx < 0:
            print(f"  Warning: Invalid face indices detected (range: {min_face_idx} to {max_face_idx}, vertices: {num_vertices})")
            valid_faces_mask = np.all(mesh.faces < num_vertices, axis=1) & np.all(mesh.faces >= 0, axis=1)
            mesh.update_faces(valid_faces_mask)
            print(f"  Removed invalid faces -> {len(mesh.faces)} faces remaining")

    # Step 1: Remove duplicate faces using the non-deprecated API.
    try:
        unique_faces = mesh.unique_faces()
        if len(unique_faces) != len(mesh.faces):
            mesh.update_faces(unique_faces)
            mesh.remove_unreferenced_vertices()
            print(f"  Removed duplicate faces -> {len(mesh.faces)} faces")
    except Exception as e:
        print(f"  Warning: Failed to remove duplicate faces: {e}")

    # Step 2: Drop degenerate faces that collapse to a line or point.
    try:
        nondegenerate_mask = mesh.nondegenerate_faces()
        if not np.all(nondegenerate_mask):
            mesh.update_faces(nondegenerate_mask)
            mesh.remove_unreferenced_vertices()
            print(f"  Removed degenerate faces -> {len(mesh.faces)} faces")
    except Exception as e:
        print(f"  Warning: Failed to remove degenerate faces: {e}")

    # Step 3: Split into connected components and keep the one with largest surface area.
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            print(f"  Found {len(components)} connected components")
            components = sorted(components, key=lambda m: (m.area, len(m.faces)), reverse=True)
            mesh = components[0]
            print(f"  Retained largest component: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    except Exception as e:
        print(f"  Warning: Failed to split components: {e}")

    # Step 4: Repair holes and close small gaps to approach watertightness.
    try:
        if not mesh.is_watertight:
            print("  Mesh is not watertight, attempting repairs...")

            changed = mesh.fill_holes()
            if changed:
                print("    Filled holes via fill_holes()")

            if not mesh.is_watertight:
                mesh.remove_unreferenced_vertices()
                mesh.merge_vertices()
                mesh.remove_unreferenced_vertices()
                mesh.fill_holes()
    except Exception as e:
        print(f"  Warning: Failed to repair mesh watertightness: {e}")

    # Step 5: Final cleanup and validation to guarantee consistency.
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    
    # Step 6: Validate face indices to prevent segmentation faults during export
    # Remove faces with invalid vertex indices
    num_vertices = len(mesh.vertices)
    if len(mesh.faces) > 0:
        valid_faces_mask = np.all(mesh.faces < num_vertices, axis=1) & np.all(mesh.faces >= 0, axis=1)
        
        if not np.all(valid_faces_mask):
            invalid_count = (~valid_faces_mask).sum()
            print(f"  Warning: Found {invalid_count} faces with invalid vertex indices, removing...")
            mesh.update_faces(valid_faces_mask)
            mesh.remove_unreferenced_vertices()
            print(f"  Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 7: Ensure data types are correct (int32 for faces, float for vertices)
    try:
        if mesh.faces.dtype != np.int32 and mesh.faces.dtype != np.int64:
            print(f"  Converting faces from {mesh.faces.dtype} to int32...")
            mesh.faces = mesh.faces.astype(np.int32)
        
        if not np.issubdtype(mesh.vertices.dtype, np.floating):
            print(f"  Converting vertices from {mesh.vertices.dtype} to float32...")
            mesh.vertices = mesh.vertices.astype(np.float32)
    except Exception as e:
        print(f"  Warning: Failed to fix data types: {e}")

    is_watertight = mesh.is_watertight
    is_winding_consistent = mesh.is_winding_consistent

    print(f"  Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    print(f"  Watertight: {is_watertight}, Winding consistent: {is_winding_consistent}")

    if not is_watertight:
        print("  Warning: Mesh could not be made fully watertight, but has been cleaned")

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
    image_processor = ImageProcessorV2(1024, border_ratio=0.1)
    pipeline.image_processor = image_processor
    
    with Image.open(matted_image_path) as image_pil:
        processed_result = image_processor(image_pil)
        processed_img = processed_result['image']

    processed_img_path = preprocessed_img_dir / f'{image_name}.png'
    processed_img_pil = torch_to_pil(processed_img)
    processed_img_pil.save(str(processed_img_path))
    
    # Setup generator
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
   
    # Check for differentiable marching cubes support
    if False and _HAS_DISO:
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
        generator=generator,
        output_type='trimesh',
        enable_pbar=True,
        # guidance_scale=5.0,
        num_inference_steps=30,
        octree_resolution=512,
        mc_algo=mc_algo,
        num_chunks=8000,
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
    # mesh = post_process_mesh(mesh)
    
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
def run_texture_inference(image_path, mesh_path, output_dir, max_num_view=6, resolution=768):
    output_dir = Path(output_dir)
    image_path = Path(image_path)
    mesh_path = Path(mesh_path)

    image_name = image_path.stem
    textured_mesh_dir = output_dir / 'textured_mesh' / image_name
    textured_mesh_dir.mkdir(parents=True, exist_ok=True)

    # Initialize paint pipeline
    print("Loading Hunyuan3D paint pipeline...")
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.multiview_cfg_path = "/localhome/aha220/Hairdar/modules/Hunyuan3D-2.1/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "/localhome/aha220/Hairdar/modules/Hunyuan3D-2.1/hy3dpaint/hunyuanpaintpbr"
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

def run_single_inference(image_name: str, output_dir):
    """Main function to run the shape inference pipeline."""
    fname = image_name
    matted_image_path = output_dir / 'matted_image' / f'{fname}.png'

    if not matted_image_path.is_file():
        raise FileNotFoundError(f"Matted image not found: {matted_image_path}")

    mesh_path = output_dir / 'shape_mesh' / f'{fname}.glb'
    textured_mesh_path = output_dir / 'textured_mesh' / fname / 'textured_mesh.glb'

    if mesh_path.exists():
        print(f"Mesh already exists at {mesh_path}, skipping shape inference.")
    else:
        run_shape_inference(
            image_path=str(matted_image_path),
            output_dir=str(output_dir),
            seed=32
        )

    if textured_mesh_path.exists():
        print(f"Textured mesh already exists at {textured_mesh_path}, skipping texture inference.")
    else:
        textured_mesh_path = run_texture_inference(
            image_path=matted_image_path,
            mesh_path=mesh_path,
            output_dir=output_dir
        )

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
    

def main(data_dir: str = "/localhome/aha220/Hairdar/modules/Hunyuan3D-2.1/outputs/"):
    output_dir = Path(data_dir)
    matted_images_dir = output_dir / 'matted_image'

    if not matted_images_dir.exists():
        raise FileNotFoundError(f"Matted images directory not found: {matted_images_dir}")

    for image_file in matted_images_dir.glob("*.png"):
        try:
            if image_file.stem != "hair_113":
                continue
            run_single_inference(
                image_name=image_file.stem,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        
if __name__ == "__main__":
    main()