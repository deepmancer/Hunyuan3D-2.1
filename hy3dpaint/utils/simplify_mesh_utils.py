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

import trimesh
import pymeshlab
import numpy as np


def remesh_mesh(mesh_path, remesh_path):
    mesh = mesh_simplify_trimesh(mesh_path, remesh_path)


def mesh_simplify_trimesh(inputpath, outputpath, target_count=200000):
    # 先去除离散面
    ms = pymeshlab.MeshSet()
    if inputpath.endswith(".glb"):
        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(inputpath)
    ms.save_current_mesh(outputpath.replace(".glb", ".obj"), save_textures=False)
    
    # 调用减面函数
    courent = trimesh.load(outputpath.replace(".glb", ".obj"), force="mesh")
    
    # Validate mesh before processing
    print(f"Loaded mesh for remeshing: {len(courent.vertices)} vertices, {len(courent.faces)} faces")
    
    # Check for invalid face indices
    num_vertices = len(courent.vertices)
    if len(courent.faces) > 0:
        max_face_idx = courent.faces.max()
        min_face_idx = courent.faces.min()
        
        if max_face_idx >= num_vertices or min_face_idx < 0:
            print(f"Warning: Invalid face indices in loaded mesh (range: {min_face_idx} to {max_face_idx}, vertices: {num_vertices})")
            print(f"Cleaning mesh before remeshing...")
            valid_faces_mask = np.all(courent.faces < num_vertices, axis=1) & np.all(courent.faces >= 0, axis=1)
            courent.update_faces(valid_faces_mask)
            courent.remove_unreferenced_vertices()
            print(f"Cleaned mesh: {len(courent.vertices)} vertices, {len(courent.faces)} faces")
    
    face_num = courent.faces.shape[0]

    if face_num > target_count:
        print(f"Simplifying mesh from {face_num} to {target_count} faces...")
        courent = courent.simplify_quadric_decimation(target_count)
        print(f"Simplified mesh: {len(courent.vertices)} vertices, {len(courent.faces)} faces")
        
        # Validate again after simplification
        num_vertices = len(courent.vertices)
        if len(courent.faces) > 0:
            max_face_idx = courent.faces.max()
            min_face_idx = courent.faces.min()
            
            if max_face_idx >= num_vertices or min_face_idx < 0:
                print(f"Warning: Simplification produced invalid face indices, cleaning...")
                valid_faces_mask = np.all(courent.faces < num_vertices, axis=1) & np.all(courent.faces >= 0, axis=1)
                courent.update_faces(valid_faces_mask)
                courent.remove_unreferenced_vertices()
                print(f"Cleaned simplified mesh: {len(courent.vertices)} vertices, {len(courent.faces)} faces")
    
    # Final validation before export
    if len(courent.faces) > 0:
        # Ensure correct data type
        if courent.faces.dtype not in (np.int32, np.int64):
            print(f"Warning: Converting face indices from {courent.faces.dtype} to int32 before export")
            courent.faces = np.clip(courent.faces, 0, len(courent.vertices) - 1).astype(np.int32)
    
    courent.export(outputpath)
    print(f"Exported remeshed mesh to: {outputpath}")
