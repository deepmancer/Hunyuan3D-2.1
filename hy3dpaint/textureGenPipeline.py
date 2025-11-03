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

import os
import torch
import bpy
import math
import copy
import trimesh
import numpy as np
from PIL import Image
from collections.abc import Sequence
from mathutils import Matrix
from DifferentiableRenderer.MeshRender import MeshRender


import trimesh
import pymeshlab

from utils.simplify_mesh_utils import remesh_mesh
from utils.multiview_utils import multiviewDiffusionNet
from utils.pipeline_utils import ViewProcessor
from utils.image_super_utils import imageSuperNet
from utils.uvwrap_utils import mesh_uv_wrap
import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


def apply_subdivision_surface(mesh_obj, levels=2, render_levels=2, subdivision_type='CATMULL_CLARK'):
    if mesh_obj.type != 'MESH':
        return

    # Clamp subdivision levels to valid range
    levels = max(0, int(levels))
    render_levels = max(0, int(render_levels))

    # Add subdivision surface modifier
    modifier = mesh_obj.modifiers.new(name="Subdivision", type='SUBSURF')
    modifier.levels = levels
    modifier.render_levels = render_levels
    modifier.subdivision_type = subdivision_type
    
    # Use the limit surface for smoother results
    modifier.show_only_control_edges = True
    
    # Enable "Use Custom Normals" - preserves custom normals during subdivision
    modifier.use_custom_normals = True
    
    print(
        f"Applied Subdivision Surface modifier on {mesh_obj.name} with {levels} viewport levels "
        f"and {render_levels} render levels (custom normals: {modifier.use_custom_normals})"
    )


def apply_triangulate_modifier(mesh_obj, quad_method='BEAUTY', ngon_method='BEAUTY'):
    if mesh_obj.type != 'MESH':
        return

    # Add triangulate modifier
    modifier = mesh_obj.modifiers.new(name="Triangulate", type='TRIANGULATE')
    modifier.quad_method = quad_method
    modifier.ngon_method = ngon_method
    modifier.keep_custom_normals = True  # Preserve custom normals from subdivision
    
    print(
        f"Applied Triangulate modifier on {mesh_obj.name} with quad_method={quad_method}, "
        f"ngon_method={ngon_method}"
    )


def apply_smooth_shading(mesh_obj, auto_smooth_angle=math.pi):
    if mesh_obj.type != 'MESH':
        return
    mesh_data = mesh_obj.data
    if mesh_data is None or not mesh_data.polygons:
        return

    for poly in mesh_data.polygons:
        poly.use_smooth = True

    # Enable auto smooth for better normal calculation
    mesh_data.use_auto_smooth = True
    mesh_data.auto_smooth_angle = auto_smooth_angle
    
    print(
        f"Applied smooth shading with auto smooth angle {math.degrees(auto_smooth_angle):.1f}° on {mesh_obj.name}"
    )

# ----------------------------
# Helpers
# ----------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Ensure glTF exporter is available (it ships with Blender)
    if 'io_scene_gltf2' not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module='io_scene_gltf2')

def import_mesh(mesh_path: str):
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext in [".fbx"]:
        bpy.ops.wm.fbx_import(filepath=mesh_path)
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext in [".ply"]:
        bpy.ops.wm.ply_import(filepath=mesh_path)
    elif ext in [".stl"]:
        bpy.ops.wm.stl_import(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No mesh objects found after import.")
    return mesh_objs

def ensure_uv_map(obj, uv_map_name=None):
    if not obj.data.uv_layers:
        raise RuntimeError(f"Object '{obj.name}' has no UV layers.")
    if uv_map_name:
        if uv_map_name not in obj.data.uv_layers:
            raise RuntimeError(
                f"UV map '{uv_map_name}' not found on '{obj.name}'. "
                f"Available: {[uv.name for uv in obj.data.uv_layers]}"
            )
        obj.data.uv_layers.active = obj.data.uv_layers[uv_map_name]
    else:
        # Use whatever is already active
        pass
    return obj.data.uv_layers.active.name

def load_image_node(nodes, image_path, label, colorspace="sRGB"):
    node = nodes.new("ShaderNodeTexImage")
    node.label = label
    node.interpolation = 'Smart'
    img = bpy.data.images.load(image_path)
    node.image = img
    # Color space: albedo = sRGB; data maps = Non-Color
    node.image.colorspace_settings.name = colorspace
    return node

def build_pbr_material(mat_name, base_color_path, roughness_path, metallic_path, uv_map_name=None):
    # Create material
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    # Clear default nodes except output
    for n in list(nodes):
        if n.type != 'OUTPUT_MATERIAL':
            nodes.remove(n)
    out = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL')

    # Principled BSDF
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (200, 200)

    # UV Map node (optional—if omitted, Image Texture nodes use the active UV)
    uvnode = nodes.new("ShaderNodeUVMap")
    uvnode.from_instancer = False
    if uv_map_name:
        uvnode.uv_map = uv_map_name
    uvnode.location = (-900, 200)

    # Texture nodes
    tex_base = load_image_node(nodes, base_color_path, "BaseColor", colorspace="sRGB")
    tex_base.location = (-650, 300)

    tex_rough = load_image_node(nodes, roughness_path, "Roughness", colorspace="Non-Color")
    tex_rough.location = (-650, 50)

    tex_metal = load_image_node(nodes, metallic_path, "Metallic", colorspace="Non-Color")
    tex_metal.location = (-650, -200)

    # Wire UVs to textures
    links.new(uvnode.outputs["UV"], tex_base.inputs["Vector"])
    links.new(uvnode.outputs["UV"], tex_rough.inputs["Vector"])
    links.new(uvnode.outputs["UV"], tex_metal.inputs["Vector"])

    # Wire textures to Principled
    links.new(tex_base.outputs["Color"], principled.inputs["Base Color"])
    links.new(tex_rough.outputs["Color"], principled.inputs["Roughness"])
    links.new(tex_metal.outputs["Color"], principled.inputs["Metallic"])

    # Alpha handling (optional): if base color has alpha you want to use
    # principled.inputs["Alpha"].default_value = 1.0
    # out->Surface
    links.new(principled.outputs["BSDF"], out.inputs["Surface"])

    # Ensure glTF-compatible settings
    mat.blend_method = 'OPAQUE'  # set 'CLIP'/'HASHED' if you actually use alpha
    return mat

def assign_material(objs, mat):
    for o in objs:
        if o.type != 'MESH':
            continue
        if not o.data.materials:
            o.data.materials.append(mat)
        else:
            # Replace all existing slots with our mat
            for i in range(len(o.data.materials)):
                o.data.materials[i] = mat

def export_glb(filepath, select_objs=None):
    if select_objs:
        # Select only the provided objects
        bpy.ops.object.select_all(action='DESELECT')
        for o in select_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = select_objs[0]
        use_selection = True
    else:
        use_selection = False
        select_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']

    # # Rotate meshes so Blender's forward (+Y) / up (+Z) map to desired forward (-Z) / up (+Y)
    # rotation = Matrix.Rotation(math.radians(-90.0), 4, 'X')
    # original_transforms = {obj: obj.matrix_world.copy() for obj in select_objs}
    # for obj in select_objs:
    #     obj.matrix_world = rotation @ obj.matrix_world

    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        export_materials='EXPORT',
        export_texcoords=True,
        export_normals=True,
        export_cameras=False,
        export_lights=False,
        export_yup=True,
        export_image_format='AUTO',
        use_selection=use_selection,
        export_apply=True
    )

    # Restore original transforms to keep scene state predictable for subsequent operations
    # for obj, original_matrix in original_transforms.items():
    #     obj.matrix_world = original_matrix


def add_texture_to_mesh(
    mesh_path,
    base_color_path,
    metallic_path,
    roughness_path,
    uv_map_name=None,
    out_path="./textured.glb",
    subdivision_levels=2,
    subdivision_render_levels=3,
    subdivision_type="SIMPLE",
    auto_smooth_angle=30.0,
    triangulate_quad_method="BEAUTY",
    triangulate_ngon_method="BEAUTY"
):
    reset_scene()

    mesh_objs = import_mesh(os.path.abspath(mesh_path))

    # Ensure the chosen UV map is active on each mesh object
    active_uv_name = None
    smooth_angle_deg = max(0.0, min(180.0, auto_smooth_angle))
    smooth_angle_rad = math.radians(smooth_angle_deg)
    for obj in mesh_objs:
        active_uv_name = ensure_uv_map(obj, uv_map_name)  # validates and (optionally) sets active
        apply_subdivision_surface(
            obj,
            levels=subdivision_levels,
            render_levels=subdivision_render_levels,
            subdivision_type=subdivision_type,
        )
        # apply_triangulate_modifier(
        #     obj,
        #     quad_method=triangulate_quad_method,
        #     ngon_method=triangulate_ngon_method,
        # )
        apply_smooth_shading(obj, auto_smooth_angle=smooth_angle_rad)
    # Build material (use the validated/active UV name)
    mat = build_pbr_material(
        mat_name="PBR_Material",
        base_color_path=os.path.abspath(base_color_path),
        roughness_path=os.path.abspath(roughness_path),
        metallic_path=os.path.abspath(metallic_path),
        uv_map_name=active_uv_name
    )
    assign_material(mesh_objs, mat)

    # Export GLB
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_glb(out_path, select_objs=mesh_objs)

    print(f"[OK] Exported: {out_path}")

class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view, resolution):
        self.device = "cuda"

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"

        # view selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipeline:

    def __init__(self, config=None) -> None:
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.models = {}
        self.stats_logs = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        self.load_models()

    def load_models(self):
        torch.cuda.empty_cache()
        self.models["super_model"] = imageSuperNet(self.config)
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        print("Models Loaded.")

    @torch.no_grad()
    def __call__(
        self,
        mesh_path=None,
        image_path=None,
        output_mesh_path=None,
        use_remesh=True,
        save_glb=False,
    ):
        """Generate texture for 3D mesh using multiview diffusion."""

        if mesh_path is None:
            raise ValueError("mesh_path must be provided")

        mesh_path = os.fspath(mesh_path)
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        if image_path is None:
            raise ValueError("image_path must be provided")

        # Normalize image inputs into a list of PIL images
        image_prompt = []
        candidates = image_path if isinstance(image_path, Sequence) and not isinstance(image_path, (str, bytes)) else [image_path]
        for candidate in candidates:
            if isinstance(candidate, Image.Image):
                image_prompt.append(candidate)
            elif isinstance(candidate, (str, os.PathLike)):
                if not os.path.exists(candidate):
                    raise FileNotFoundError(f"Reference image not found: {candidate}")
                image_prompt.append(Image.open(candidate))
            else:
                raise TypeError(f"Unsupported image input type: {type(candidate)}")

        if not image_prompt:
            raise ValueError("No valid images were provided")

        # Process mesh
        path = os.path.dirname(mesh_path)
        sample_name = os.path.splitext(os.path.basename(mesh_path))[0]
        if use_remesh:
            processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
            remesh_mesh(mesh_path, processed_mesh_path)
        else:
            processed_mesh_path = mesh_path

        if not os.path.exists(processed_mesh_path):
            raise FileNotFoundError(f"Processed mesh file not found: {processed_mesh_path}")

        # Output path
        if output_mesh_path is None:
            base_path = os.path.join(path, "textured_mesh")
            target_ext = ".ply"
        else:
            base_path, target_ext = os.path.splitext(os.fspath(output_mesh_path))
            if target_ext == "":
                target_ext = ".ply"
            if target_ext.lower() not in (".ply", ".obj", ".glb"):
                raise ValueError(f"Unsupported output extension: {target_ext}")

        ply_output_path = f"{base_path}.ply"
        glb_output_path = f"{base_path}.glb"

        # Load mesh
        mesh = trimesh.load(processed_mesh_path, force="mesh")
        
        # Validate and clean mesh immediately after loading
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Check for invalid face indices that could cause segfaults
        num_vertices = len(mesh.vertices)
        if len(mesh.faces) > 0:
            # Check for out-of-range indices
            max_face_idx = mesh.faces.max()
            min_face_idx = mesh.faces.min()
            
            if max_face_idx >= num_vertices or min_face_idx < 0:
                print(f"Warning: Invalid face indices detected (range: {min_face_idx} to {max_face_idx}, vertices: {num_vertices})")
                print(f"Cleaning mesh to remove invalid faces...")
                
                # Remove faces with invalid indices
                valid_faces_mask = np.all(mesh.faces < num_vertices, axis=1) & np.all(mesh.faces >= 0, axis=1)
                mesh.update_faces(valid_faces_mask)
                mesh.remove_unreferenced_vertices()
                print(f"Cleaned mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Ensure face indices are int32 (not uint32 or corrupted types)
        if len(mesh.faces) > 0:
            if mesh.faces.dtype not in (np.int32, np.int64):
                print(f"Warning: Converting face indices from {mesh.faces.dtype} to int32")
                # First clip any out-of-range values before converting
                mesh.faces = np.clip(mesh.faces, 0, len(mesh.vertices) - 1)
                mesh.faces = mesh.faces.astype(np.int32)
        
        # Validate vertices are finite
        if not np.all(np.isfinite(mesh.vertices)):
            print("Warning: Mesh contains non-finite vertices, attempting to fix...")
            finite_mask = np.all(np.isfinite(mesh.vertices), axis=1)
            if not np.all(finite_mask):
                # This is a severe issue - we can't safely proceed
                raise ValueError(f"Mesh contains {(~finite_mask).sum()} non-finite vertices that cannot be repaired")
        
        print("Mesh validation complete, proceeding with UV wrapping...")
        mesh = mesh_uv_wrap(mesh)
        self.render.load_mesh(mesh=mesh)

        ########### View Selection #########
        selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num,
        )

        normal_maps = self.view_processor.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True
        )
        position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)

        ##########  Style  ###########
        image_caption = "high quality"
        image_style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white_bg = Image.new("RGB", image.size, (255, 255, 255))
                white_bg.paste(image, mask=image.getchannel("A"))
                image = white_bg
            image_style.append(image)
        image_style = [image.convert("RGB") for image in image_style]

        ###########  Multiview  ##########
        multiviews_pbr = self.models["multiview_model"](
            image_style,
            normal_maps + position_maps,
            prompt=image_caption,
            custom_view_size=self.config.resolution,
            resize_input=True,
        )
        ###########  Enhance  ##########
        enhance_images = {}
        enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
        enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])

        for i in range(len(enhance_images["albedo"])):
            enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
            enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])

        ###########  Bake  ##########
        num_views = len(enhance_images["albedo"])
        for i in range(num_views):
            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                (self.config.render_size, self.config.render_size)
            )
            enhance_images["mr"][i] = enhance_images["mr"][i].resize((self.config.render_size, self.config.render_size))
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture_mr, mask_mr = self.view_processor.bake_from_multiview(
            enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        ##########  inpaint  ###########
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self.render.set_texture_mr(texture_mr)

        export_ext = target_ext.lower()
        mesh_export_path = {
            ".ply": ply_output_path,
        }[".ply"]

        texture_maps = self.render.save_mesh(mesh_export_path, downsample=True)

        result_path = mesh_export_path if export_ext != ".glb" else glb_output_path

        generate_glb = save_glb or export_ext == ".glb"
        if generate_glb:
            texture_dir = os.path.dirname(mesh_export_path)

            def _resolve_texture_path(texture_key, friendly_name):
                if texture_maps is None or texture_key not in texture_maps:
                    raise RuntimeError(f"Missing {friendly_name} texture map from mesh export (key='{texture_key}').")

                texture_path = os.path.join(texture_dir, texture_maps[texture_key])
                if not os.path.exists(texture_path):
                    raise FileNotFoundError(f"Expected {friendly_name} texture map not found: {texture_path}")

                return texture_path

            albedo_path = _resolve_texture_path("diffuse", "albedo")
            metallic_path = _resolve_texture_path("metallic", "metallic")
            roughness_path = _resolve_texture_path("roughness", "roughness")

            add_texture_to_mesh(
                mesh_path=mesh_export_path.replace(".glb", ".ply"),
                base_color_path=albedo_path,
                metallic_path=metallic_path,
                roughness_path=roughness_path,
                uv_map_name=None,
                out_path=glb_output_path.replace(".ply", ".glb"),
                subdivision_levels=1,
                subdivision_render_levels=3,
            )

        return result_path
