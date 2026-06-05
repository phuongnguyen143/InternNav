import trimesh
import numpy as np


def convert_yup_to_zup(input_path, output_path):
    # Load scene (có thể chứa nhiều mesh)
    scene = trimesh.load(input_path, force="scene")

    # Rotation matrix: -90° quanh trục X
    angle = -np.pi/2
    rot = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])

    # Apply cho toàn bộ scene
    scene.apply_transform(rot)

    # Export lại
    scene.export(output_path)

    print(f"Converted: {input_path} -> {output_path}")


if __name__ == "__main__":
    convert_yup_to_zup("/home/lenguyen1/hoangpqn/InternNav/data/scene_data/mp3d/office/5_6_2026.glb", "/home/lenguyen1/hoangpqn/InternNav/data/scene_data/mp3d/office/5_6_2026_zup.glb")