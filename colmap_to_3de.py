import json
import os
import random
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


def filter_by_min_track_length(points, min_track_length):
    """Drop points observed in fewer than min_track_length images (COLMAP's
    TRACK[] length) -- points with a short track are more likely to be noisy
    or spurious triangulations. None/0 disables filtering."""
    if not min_track_length:
        return points
    return [p for p in points if p["track_length"] >= min_track_length]


def sample_points(points, max_points, seed=None):
    """Return `points` unchanged when max_points is None/0 (no limit);
    otherwise a uniform random sample of at most max_points, without
    replacement, so the exported subset stays representative of the full
    point cloud's spatial distribution rather than favoring whatever
    POINT3D_ID ordering COLMAP happened to emit."""
    if not max_points or max_points >= len(points):
        return points
    rng = random.Random(seed)
    return rng.sample(points, max_points)


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


def write_camera_import_script(scene, scene_name, sensor_width_mm, out_path, images_dir):
    """Generate a self-contained .py script that, when run from inside a
    live 3DEqualizer session (via its Script Database menu -- see the
    3DE4.script.gui header below), creates a CAMERA point group with a fully
    animated camera: per-frame position/rotation (from colmap_matrix_to_3de),
    static focal length, film back, and principal-point offset.

    sensor_width_mm is the same physical-sensor-width-in-mm input
    build_houdini_scene.py --sensor_width_mm already requires, since the
    source JSON's fl_x/fl_y are in pixels, not a physical unit."""
    fl_x = scene["fl_x"]
    sensor_w_px = scene["sensor_w"]
    img_w = scene["w"]
    img_h = scene["h"]
    cx = scene["cx"]
    cy = scene["cy"]

    focal_mm = (fl_x / sensor_w_px) * sensor_width_mm
    focal_cm = focal_mm / 10.0

    fback_w_mm = sensor_width_mm * (img_w / sensor_w_px)
    fback_h_mm = fback_w_mm * (img_h / img_w)
    fback_w_cm = fback_w_mm / 10.0
    fback_h_cm = fback_h_mm / 10.0

    winx_frac = (img_w / 2 - cx) / img_w
    winy_frac = (img_h / 2 - cy) / img_h
    lens_center_x_cm = -winx_frac * fback_w_cm
    lens_center_y_cm = -winy_frac * fback_h_cm

    frames = scene["frames"]
    frame_nums = [f["frame_num"] for f in frames]
    start_frame = frame_nums[0]
    end_frame = frame_nums[-1]

    lines = []
    lines.append("#")
    lines.append("# 3DE4.script.name:\tImport COLMAP Camera (%s)..." % scene_name)
    lines.append("#")
    lines.append("# 3DE4.script.version:\tv1.0")
    lines.append("#")
    lines.append("# 3DE4.script.gui:\tMain Window::3DE4::File::Import")
    lines.append("#")
    lines.append("# 3DE4.script.comment:\tImports the animated camera solved by the "
                  "AI Colmap Camera Tracking pipeline for scene '%s'." % scene_name)
    lines.append("#")
    lines.append("")
    lines.append("import tde4")
    lines.append("")
    lines.append("pgroup_id = tde4.createPGroup(\"CAMERA\")")
    lines.append("tde4.setPGroupName(pgroup_id, \"%s\")" % scene_name)
    lines.append("camera_id = tde4.createCamera(\"SEQUENCE\")")
    lines.append("tde4.setCameraName(camera_id, \"%s\")" % scene_name)
    lines.append("tde4.setCameraImageWidth(camera_id, %d)" % int(img_w))
    lines.append("tde4.setCameraImageHeight(camera_id, %d)" % int(img_h))
    lines.append("tde4.setCameraPath(camera_id, \"%s/frame_######.jpg\")"
                  % images_dir.replace(os.sep, "/"))
    lines.append("tde4.setCameraSequenceAttr(camera_id, %d, %d, 1)"
                  % (start_frame, end_frame))
    lines.append("")
    lines.append("lens_id = tde4.createLens()")
    lines.append("tde4.setLensName(lens_id, \"%s_lens\")" % scene_name)
    lines.append("tde4.setLensFBackWidth(lens_id, %s)" % repr(round(fback_w_cm, 6)))
    lines.append("tde4.setLensFBackHeight(lens_id, %s)" % repr(round(fback_h_cm, 6)))
    lines.append("tde4.setLensPixelAspect(lens_id, 1.0)")
    lines.append("tde4.setLensLensCenterX(lens_id, %s)" % repr(round(lens_center_x_cm, 6) + 0.0))
    lines.append("tde4.setLensLensCenterY(lens_id, %s)" % repr(round(lens_center_y_cm, 6) + 0.0))
    lines.append("tde4.setCameraLens(camera_id, lens_id)")
    lines.append("")

    for frame_data in frames:
        frame_num = frame_data["frame_num"]
        position, rotation = colmap_matrix_to_3de(frame_data["transform_matrix"])
        lines.append("tde4.setCameraFocalLength(camera_id, %d, %s)"
                      % (frame_num, repr(round(focal_cm, 6))))
        lines.append("tde4.setPGroupPosition3D(pgroup_id, camera_id, %d, %s)"
                      % (frame_num, [round(v, 9) for v in position]))
        lines.append("tde4.setPGroupRotation3D(pgroup_id, camera_id, %d, %s)"
                      % (frame_num, [[round(v, 9) for v in row] for row in rotation]))

    lines.append("")
    # Make the newly created pgroup/camera current so 3DE's UI (timeline,
    # 3D viewer) and native features that operate on the current pgroup
    # (e.g. File > Import > Import Survey Textfile...) target this camera.
    lines.append("tde4.setCurrentPGroup(pgroup_id)")
    lines.append("tde4.setCurrentCamera(camera_id)")
    lines.append("")
    lines.append("tde4.postQuestionRequester(\"Import COLMAP Camera...\", "
                  "\"Camera '%s' imported successfully. Use File > Import > "
                  "Import Survey Textfile... to import the point cloud.\", \"Ok\")" % scene_name)
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


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
            # Fixed fields are ID,X,Y,Z,R,G,B,ERROR (8); everything after
            # that is TRACK[] as repeated (IMAGE_ID, POINT2D_IDX) pairs --
            # its pair count is how many images this point was observed in.
            points.append({
                "id": parts[0],
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "track_length": (len(parts) - 8) // 2,
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
