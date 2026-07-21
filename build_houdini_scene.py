import hou
import argparse
import json
import os
import re
import sys

_DISTORTION_KEYS = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")


def _add_distortion_reference_parms(cam, data, fl_x, fl_y, cx, cy):
    """Add a read-only 'OpenCV Distortion' spare-parameter folder to `cam`
    when the JSON carries a camera_model and OpenCV distortion coefficients
    (written by the --keep-distortion path). No lens shader or VOP network
    is created -- these values are for the user to copy into their own
    kma_physicallens node. No-op when the fields aren't present (rectified
    JSON, or JSON produced before this field existed)."""
    camera_model = data.get("camera_model")
    if not camera_model or not any(float(data.get(key, 0.0)) != 0.0 for key in _DISTORTION_KEYS):
        return

    folder_parms = [
        hou.StringParmTemplate("cv_camera_model", "Camera Model", 1, default_value=(camera_model,)),
        hou.FloatParmTemplate("cv_fx", "fx (px)", 1, default_value=(fl_x,)),
        hou.FloatParmTemplate("cv_fy", "fy (px)", 1, default_value=(fl_y,)),
        hou.FloatParmTemplate("cv_cx", "cx (px)", 1, default_value=(cx,)),
        hou.FloatParmTemplate("cv_cy", "cy (px)", 1, default_value=(cy,)),
    ]
    for key in _DISTORTION_KEYS:
        folder_parms.append(
            hou.FloatParmTemplate(f"cv_{key}", key, 1, default_value=(float(data.get(key, 0.0)),))
        )
    for pt in folder_parms:
        pt.setConditional(hou.parmCondType.DisableWhen, "{ 1 == 1 }")

    folder = hou.FolderParmTemplate(
        "opencv_distortion_folder", "OpenCV Distortion", folder_parms,
        folder_type=hou.folderType.Simple,
    )

    group = cam.parmTemplateGroup()
    group.append(folder)
    cam.setParmTemplateGroup(group)


def create_animated_camera(json_path, aperture_width=36.0):
    cam_name = "Nerfstudio_Animated_Cam"

    # 1. Check file
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON file not found: {json_path}")
        return

    print(f"Loading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    json_dir = os.path.dirname(json_path)

    # 2. Get basic information
    frames = data.get("frames", [])

    # Sort by number in filename (ensure correct animation order)
    def get_frame_num(frame_data):
        fname = os.path.basename(frame_data['file_path'])
        match = re.search(r'(\d+)', fname)
        return int(match.group(1)) if match else 0

    frames.sort(key=get_frame_num)

    if not frames:
        print("No frames found in JSON.")
        return

    # Background image sequence path, derived from the first frame's own
    # file_path rather than a hardcoded folder name -- this works whether
    # frames point into images_undistorted/ (rectify path) or back to the
    # original images/ folder (--keep-distortion path).
    first_frame_path = frames[0]["file_path"]
    frame_dir = os.path.dirname(first_frame_path)
    frame_name = os.path.basename(first_frame_path)
    frame_pattern = re.sub(r'\d+', '$F6', frame_name, count=1)
    background_image_path = os.path.abspath(
        os.path.join(json_dir, frame_dir, frame_pattern)
    ).replace(os.sep, '/')

    # Read resolution and focal length
    img_w = float(data.get("w", 1920))
    img_h = float(data.get("h", 1080))
    fl_x  = float(data.get("fl_x", 1000))  # Focal length in pixels
    fl_y  = float(data.get("fl_y", fl_x))
    cx    = float(data.get("cx", img_w / 2))
    cy    = float(data.get("cy", img_h / 2))

    # sensor_w/sensor_h: original sensor dimensions before canvas expansion.
    # Written by undistortionNerfstudioColmap.py when the undistorted canvas is
    # larger than the original image.  Falls back to img_w/img_h for older JSON
    # files (where no expansion occurred).
    sensor_w = float(data.get("sensor_w", img_w))

    # Physical focal length -- must be derived from the original sensor width,
    # not the (potentially expanded) canvas width, so the mm value is stable.
    focal_mm = (fl_x / sensor_w) * aperture_width

    # Scale aperture to match the expanded canvas so that Houdini's
    # focal/aperture ratio correctly represents the wider field of view.
    # When sensor_w == img_w (no expansion), aperture_effective == aperture_width.
    aperture_effective = aperture_width * (img_w / sensor_w)

    # Principal-point offset expressed as a fraction of the canvas width.
    # Houdini winx/winy shift the projection window; 0 = centred.
    # winx > 0 -> window centre moves left  (principal point left of centre)
    # winy > 0 -> window centre moves up    (principal point above centre)
    winx = (img_w / 2 - cx) / img_w
    winy = (img_h / 2 - cy) / img_w   # note: divided by img_w, same unit as winx

    # Convert COLMAP/OpenCV world into Houdini world. colmap2nerf already put the
    # camera basis in the OpenGL convention (Y up, Z back) via M @ diag(1,-1,-1,1),
    # even under --keep_colmap_coords, but the world frame is still COLMAP's.
    #
    # The correct world conversion is a *proper rotation* (det = +1) applied
    # identically to the camera and the point cloud -- Rx(180) = diag(1,-1,-1),
    # i.e. flip both Y and Z. This is the standard COLMAP/OpenCV -> OpenGL/Houdini
    # rigid transform (Blender's COLMAP importer uses the same).
    #
    # A reflection like a Y-only flip (det = -1) cannot work: it mirrors the
    # rendered image, and no camera-basis tweak fixes it -- you can only trade an
    # upside-down image for a left/right-mirrored one. Flipping Y *and* Z keeps
    # the determinant +1, so the camera stays a valid rotation and the view is
    # both upright and un-mirrored. (Sanity check: an identity COLMAP camera,
    # M = diag(1,-1,-1,1), maps to the Houdini identity camera -- looks -Z, up +Y.)
    flip_yz = hou.Matrix4((
        (1,  0,  0, 0),
        (0, -1,  0, 0),
        (0,  0, -1, 0),
        (0,  0,  0, 1),
    ))

    # 3. Create Houdini nodes
    obj = hou.node("/obj")
    subnet = obj.node("NeRF_Import")
    if not subnet:
        subnet = obj.createNode("subnet", "NeRF_Import")

    # Create camera (destroy and recreate if it already exists)
    cam = subnet.node(cam_name)
    if cam:
        cam.destroy()
    cam = subnet.createNode("cam", cam_name)

    print(f"Creating animation for {len(frames)} frames...")

    # Set static camera parameters
    cam.parm("resx").set(img_w)
    cam.parm("resy").set(img_h)
    cam.parm("aperture").set(aperture_effective)
    cam.parm("focal").set(focal_mm)
    cam.parm("winx").set(winx)
    cam.parm("winy").set(winy)
    _add_distortion_reference_parms(cam, data, fl_x, fl_y, cx, cy)
    cam.parm("iconscale").set(0.5)

    # Set background image for viewport
    cam.parm("vm_background").set(background_image_path)

    # 4. Process animation keyframes
    with hou.undos.group("Import Nerfstudio Camera"):

        for frame_data in frames:
            # Get Frame Number
            f_num = get_frame_num(frame_data)

            # Read matrix
            raw_mtx = frame_data.get("transform_matrix")
            if raw_mtx is None:
                print(f"Warning: frame {f_num} has no transform_matrix; skipping.")
                continue

            if isinstance(raw_mtx[0], list):
                flat_mtx = [item for sublist in raw_mtx for item in sublist]
            else:
                flat_mtx = raw_mtx

            # Convert to Houdini Matrix4, transpose (Column-Major -> Row-Major),
            # then apply the COLMAP -> Houdini world rotation (flip Y and Z).
            h_mtx = hou.Matrix4(tuple(flat_mtx)).transposed() * flip_yz

            # Extract transform data
            tra = h_mtx.extractTranslates()
            rot = h_mtx.extractRotates()

            # Prepare values
            tx, ty, tz = tra[0], tra[1], tra[2]
            rx, ry, rz = rot

            # Set Keyframes
            target_parms = ["tx", "ty", "tz", "rx", "ry", "rz"]
            values = [tx, ty, tz, rx, ry, rz]

            for p_name, val in zip(target_parms, values):
                k = hou.Keyframe()
                k.setFrame(f_num)
                k.setValue(val)
                k.setExpression("linear()") 
                
                cam.parm(p_name).setKeyframe(k)

    # 5. Set scene range
    start_frame = get_frame_num(frames[0])
    end_frame = get_frame_num(frames[-1])
    
    hou.playbar.setFrameRange(start_frame, end_frame)
    hou.playbar.setPlaybackRange(start_frame, end_frame)
    hou.setFrame(start_frame)

    subnet.layoutChildren()
    cam.parm('vm_bgenable').set(1)
    cam.parm('vm_background').set(background_image_path)
    cam.setInput(0, cam.parent().indirectInputs()[0])
    print(f"Success! Animated camera created at: {cam.path()}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Houdini scene from Nerfstudio transforms JSON")
    parser.add_argument("json_path",         help="Path to transforms_undistorted.json")
    parser.add_argument("point_cloud_path",  help="Path to points3D.ply")
    parser.add_argument("output_hip_path",   help="Path for the output .hip file")
    parser.add_argument("--sensor_width_mm", type=float, default=36.0,
                        help="Physical sensor width in mm (default: 36.0 full-frame). "
                             "Common values: ARRI LF=36.7, Super35=24.89, MFT=17.3")
    args = parser.parse_args()

    json_path        = os.path.abspath(args.json_path)
    point_cloud_path = os.path.abspath(args.point_cloud_path)
    output_hip_path  = os.path.abspath(args.output_hip_path)

    create_animated_camera(json_path=json_path, aperture_width=args.sensor_width_mm)

    # Place the point cloud geo inside the same NeRF_Import subnet as the camera,
    # destroying any prior copy so re-runs don't accumulate Scene1, Scene2, ...
    # If the subnet is missing, create_animated_camera bailed out early and there
    # is nothing meaningful to save.
    subnet = hou.node("/obj/NeRF_Import")
    if subnet is None:
        print("[ERROR] /obj/NeRF_Import not found -- camera import failed; skipping scene save.")
        sys.exit(1)
    existing_scene = subnet.node("Scene")
    if existing_scene:
        existing_scene.destroy()
    scene = subnet.createNode('geo', 'Scene')
    file_node = scene.createNode('file', 'Import_Point_Cloud')
    file_node.parm('file').set(point_cloud_path)
    # COLMAP -> Houdini world rotation (flip Y and Z), matching the camera
    # matrix above. Scale (1,-1,-1) is Rx(180), a proper rotation, so the point
    # cloud and camera stay consistent and the view is un-mirrored.
    flip_node = scene.createNode('xform', 'COLMAP_to_Houdini')
    flip_node.parm('sy').set(-1)
    flip_node.parm('sz').set(-1)
    flip_node.setInput(0, file_node)
    flip_node.setDisplayFlag(True)
    flip_node.setRenderFlag(True)
    scene.setInput(0, subnet.indirectInputs()[0])
    subnet.layoutChildren()

    hou.hipFile.save(output_hip_path)
