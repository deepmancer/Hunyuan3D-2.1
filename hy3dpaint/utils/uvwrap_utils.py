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
import xatlas
import numpy as np


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if len(mesh.faces) > 500000000:
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    print(f"UV wrapping mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Validate input mesh before UV wrapping
    if len(mesh.faces) > 0:
        max_idx = mesh.faces.max()
        min_idx = mesh.faces.min()
        if max_idx >= len(mesh.vertices) or min_idx < 0:
            print(f"Warning: Input mesh has invalid face indices (range: {min_idx} to {max_idx}, vertices: {len(mesh.vertices)})")
            valid_faces_mask = np.all(mesh.faces < len(mesh.vertices), axis=1) & np.all(mesh.faces >= 0, axis=1)
            mesh.update_faces(valid_faces_mask)
            mesh.remove_unreferenced_vertices()
            print(f"Cleaned input mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    print(f"xatlas output: vmapping shape={vmapping.shape}, indices shape={indices.shape}, uvs shape={uvs.shape}")
    print(f"vmapping range: [{vmapping.min()}, {vmapping.max()}], dtype={vmapping.dtype}")
    print(f"indices range: [{indices.min()}, {indices.max()}], dtype={indices.dtype}")
    
    # Validate xatlas output
    num_new_vertices = len(mesh.vertices[vmapping])
    if len(indices) > 0:
        max_face_idx = indices.max()
        min_face_idx = indices.min()
        
        if max_face_idx >= num_new_vertices or min_face_idx < 0:
            print(f"ERROR: xatlas produced invalid face indices!")
            print(f"  Face index range: [{min_face_idx}, {max_face_idx}]")
            print(f"  Number of vertices: {num_new_vertices}")
            print(f"  This indicates a bug in xatlas or corrupted input mesh")
            
            # Try to salvage by clamping indices
            print(f"  Attempting to fix by clamping indices...")
            indices = np.clip(indices, 0, num_new_vertices - 1).astype(np.int32)
            
            # Validate after clamping
            max_after = indices.max()
            if max_after >= num_new_vertices:
                raise ValueError(f"Cannot fix xatlas output: indices still invalid after clamping")
    
    # Ensure correct data types
    indices = indices.astype(np.int32)
    vmapping = vmapping.astype(np.int32) if np.issubdtype(vmapping.dtype, np.integer) else vmapping
    
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    print(f"UV wrapped mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Final validation after UV wrapping
    if len(mesh.faces) > 0:
        max_idx = mesh.faces.max()
        if max_idx >= len(mesh.vertices):
            raise ValueError(f"UV wrapping corrupted mesh: max face index {max_idx} >= vertex count {len(mesh.vertices)}")

    return mesh
