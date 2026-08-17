import json
import os
import re


class SceneLoadError(Exception):
    """Raised when a scene folder is missing required files or has an
    unsupported shape (e.g. multi-camera)."""


def flip_point_to_3de(x, y, z):
    """Apply the same Rx(180) world-space flip build_houdini_scene.py uses
    (diag(1,-1,-1)) to a single point, in 3DE's own column-vector convention.
    3DE and Houdini share the same Y-up world (confirmed via 3DE's official
    export_houdini.py, whose Houdini-axis conversion is a documented no-op),
    so this is the same flip used for both the camera and the point cloud."""
    return (x, -y, -z)


def colmap_matrix_to_3de(matrix):
    """Convert one camera-to-world 4x4 matrix (as stored in
    transforms_undistorted.json -- column-vector convention, COLMAP/OpenCV
    axes already converted to OpenGL/NeRF camera-local axes by colmap2nerf.py,
    but still in COLMAP's original world frame) into (position, rotation) in
    3DE's column-vector convention, ready for tde4.setPGroupPosition3D /
    setPGroupRotation3D. Applies the same Rx(180) = diag(1,-1,-1) world flip
    as build_houdini_scene.py, in column-vector form (not Houdini's internal
    row-vector convention, which is the transpose of this)."""
    position = list(flip_point_to_3de(matrix[0][3], matrix[1][3], matrix[2][3]))
    rotation = [
        [matrix[0][0], matrix[0][1], matrix[0][2]],
        [-matrix[1][0], -matrix[1][1], -matrix[1][2]],
        [-matrix[2][0], -matrix[2][1], -matrix[2][2]],
    ]
    return position, rotation


def points_to_survey_lines(points):
    """Convert load_scene()'s point list into (name, x, y, z) tuples in
    3DE's flipped world space, ready for the Import Survey Textfile format."""
    lines = []
    for p in points:
        x, y, z = flip_point_to_3de(p["x"], p["y"], p["z"])
        lines.append((f"p{p['id']}", x, y, z))
    return lines


def write_survey_points_txt(points, out_path):
    """Write 3DE's native 'Import Survey Textfile' ASCII format: one
    '<name> <x> <y> <z>' line per point, whitespace-separated."""
    lines = points_to_survey_lines(points)
    with open(out_path, "w") as f:
        for name, x, y, z in lines:
            f.write(f"{name} {x:.6f} {y:.6f} {z:.6f}\n")


def _get_frame_num(file_path):
    fname = os.path.basename(file_path)
    match = re.search(r'(\d+)', fname)
    return int(match.group(1)) if match else 0


def _load_frames(data):
    frames = []
    for frame_data in data.get("frames", []):
        frames.append({
            "frame_num": _get_frame_num(frame_data["file_path"]),
            "transform_matrix": frame_data["transform_matrix"],
        })
    frames.sort(key=lambda f: f["frame_num"])
    return frames


def _load_points(points3d_path):
    points = []
    with open(points3d_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            points.append({
                "id": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
            })
    return points


def load_scene(scene_dir):
    """Read a processed COLMAP tracking scene folder (the output of
    run_autotracker.py's default, rectified pipeline) and return its camera
    intrinsics, per-frame transforms, and sparse point cloud."""
    json_path = os.path.join(scene_dir, "undistort", "transforms_undistorted.json")
    points3d_path = os.path.join(scene_dir, "sparse", "0", "points3D.txt")

    if not os.path.exists(json_path):
        raise SceneLoadError(
            f"Not found: {json_path}\n"
            "This scene must be processed by the default (rectified) pipeline "
            "-- --keep-distortion scenes are not supported."
        )
    if not os.path.exists(points3d_path):
        raise SceneLoadError(f"Not found: {points3d_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    required_keys = ("fl_x", "fl_y", "cx", "cy", "w", "h")
    if not all(k in data for k in required_keys):
        raise SceneLoadError(
            "This JSON does not have shared top-level camera intrinsics "
            "(fl_x/fl_y/cx/cy/w/h) -- multi-camera scenes are not supported."
        )

    return {
        "w": float(data["w"]), "h": float(data["h"]),
        "sensor_w": float(data.get("sensor_w", data["w"])),
        "sensor_h": float(data.get("sensor_h", data["h"])),
        "fl_x": float(data["fl_x"]), "fl_y": float(data["fl_y"]),
        "cx": float(data["cx"]), "cy": float(data["cy"]),
        "frames": _load_frames(data),
        "points": _load_points(points3d_path),
    }
