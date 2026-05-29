import hou
import argparse
import json
import os
import re
import sys

def create_animated_camera(json_path, global_scale=1, cam_name="Nerfstudio_Animated_Cam", aperture_width=36.0):
    # 1. Check file
    if not os.path.exists(json_path):
        hou.ui.displayMessage(f"Error: File not found at:\n{json_path}")
        return

    print(f"Loading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Construct path to background image sequence from json path
    json_dir = os.path.dirname(json_path)
    images_undistorted_dir = os.path.abspath(os.path.join(json_dir, "images_undistorted"))
    # Houdini uses forward slashes
    background_image_path = os.path.join(images_undistorted_dir, "frame_$F6.jpg").replace(os.sep, '/')

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

    # Read resolution and focal length
    img_w = float(data.get("w", 1920))
    img_h = float(data.get("h", 1080))
    fl_x  = float(data.get("fl_x", 1000))  # Focal length in pixels
    cx    = float(data.get("cx", img_w / 2))
    cy    = float(data.get("cy", img_h / 2))

    # sensor_w/sensor_h: original sensor dimensions before canvas expansion.
    # Written by undistortionNerfstudioColmap.py when the undistorted canvas is
    # larger than the original image.  Falls back to img_w/img_h for older JSON
    # files (where no expansion occurred).
    sensor_w = float(data.get("sensor_w", img_w))

    # Physical focal length — must be derived from the original sensor width,
    # not the (potentially expanded) canvas width, so the mm value is stable.
    focal_mm = (fl_x / sensor_w) * aperture_width

    # Scale aperture to match the expanded canvas so that Houdini's
    # focal/aperture ratio correctly represents the wider field of view.
    # When sensor_w == img_w (no expansion), aperture_effective == aperture_width.
    aperture_effective = aperture_width * (img_w / sensor_w)

    # Principal-point offset expressed as a fraction of the canvas width.
    # Houdini winx/winy shift the projection window; 0 = centred.
    # winx > 0 → window centre moves left  (principal point left of centre)
    # winy > 0 → window centre moves up    (principal point above centre)
    winx = (img_w / 2 - cx) / img_w
    winy = (img_h / 2 - cy) / img_w   # note: divided by img_w, same unit as winx

    # Map COLMAP world (Y down) into Houdini world (Y up). colmap2nerf already
    # converted the camera basis to the OpenGL convention (Y up, Z back) via
    # M @ diag(1,-1,-1,1), even under --keep_colmap_coords, but the world frame
    # is still COLMAP's. A single world Y flip on the row-major c2w matrix
    # aligns the two.
    opencv_to_houdini = hou.Matrix4((
        (1,  0, 0, 0),
        (0, -1, 0, 0),
        (0,  0, 1, 0),
        (0,  0, 0, 1),
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
    cam.parm("iconscale").set(0.5)

    # Set background image for viewport
    cam.parm("vm_background").set(background_image_path)

    # 4. Process animation keyframes
    with hou.undos.group("Import Nerfstudio Camera"):

        for frame_data in frames:
            # Get Frame Number
            f_num = get_frame_num(frame_data)

            # Read matrix
            raw_mtx = frame_data["transform_matrix"]

            if isinstance(raw_mtx[0], list):
                flat_mtx = [item for sublist in raw_mtx for item in sublist]
            else:
                flat_mtx = raw_mtx

            # Convert to Houdini Matrix4, transpose (Column-Major -> Row-Major),
            # then post-multiply by the OpenCV->Houdini world Y-flip.
            h_mtx = hou.Matrix4(tuple(flat_mtx)).transposed() * opencv_to_houdini

            # Extract transform data
            tra = h_mtx.extractTranslates()
            rot = h_mtx.extractRotates()

            # Prepare values (apply scaling)
            tx = tra[0] * global_scale
            ty = tra[1] * global_scale
            tz = tra[2] * global_scale
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

    # 6. Set scene range
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
        print("[ERROR] /obj/NeRF_Import not found — camera import failed; skipping scene save.")
        sys.exit(1)
    existing_scene = subnet.node("Scene")
    if existing_scene:
        existing_scene.destroy()
    scene = subnet.createNode('geo', 'Scene')
    file_node = scene.createNode('file', 'Import_Point_Cloud')
    file_node.parm('file').set(point_cloud_path)
    # COLMAP world Y-down -> Houdini world Y-up. Same flip applied to the
    # camera matrix above; do it here on the geo side via a Transform SOP.
    flip_y = scene.createNode('xform', 'COLMAP_to_Houdini')
    flip_y.parm('sy').set(-1)
    flip_y.setInput(0, file_node)
    flip_y.setDisplayFlag(True)
    flip_y.setRenderFlag(True)
    scene.setInput(0, subnet.indirectInputs()[0])
    subnet.layoutChildren()

    hou.hipFile.save(output_hip_path)
