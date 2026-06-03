import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # must precede `import cv2`
import sys
import subprocess
import glob
import shutil
import argparse
import json
import cv2
import numpy as np
import piexif
from tqdm import tqdm

# System Binaries (Ensure these are in your PATH)
FFMPEG = "ffmpeg"
COLMAP = "colmap"

# Top-level files with these extensions are treated as videos.
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".m4v"}

# ACEScg (AP1 primaries, D60) -> linear sRGB / Rec.709 (D65), Bradford-adapted.
# Applied to EXR frames when --acescg is set, before the sRGB transfer function.
ACEScg_TO_SRGB = np.array([
    [ 1.70505, -0.62179, -0.08326],
    [-0.13026,  1.14080, -0.01055],
    [-0.02400, -0.12897,  1.15297],
], dtype=np.float32)

def run_command(cmd, error_msg, quiet=False):
    """Runs a subprocess command. Returns True on success, False on failure."""
    try:
        kwargs = {}
        if quiet:
            kwargs['stdout'] = subprocess.DEVNULL
            kwargs['stderr'] = subprocess.DEVNULL
        
        # Run command
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, **kwargs)
        return True
    except subprocess.CalledProcessError:
        print(error_msg)
        return False
    except FileNotFoundError:
        print(f"        [ERROR] Binary not found: {cmd[0]}")
        print(error_msg)
        return False

def _patch_cameras_bin_focal_length(cameras_bin_path, fl_px):
    """Overwrite focal length in a COLMAP cameras.bin with the specified value."""
    import struct
    # model_id -> (num_params, single_f)
    # single_f=True: only params[0] is focal length
    # single_f=False: params[0]=fx, params[1]=fy
    MODEL_PARAMS = {
        0:  (3,  True),   # SIMPLE_PINHOLE
        1:  (4,  False),  # PINHOLE
        2:  (4,  True),   # SIMPLE_RADIAL
        3:  (5,  True),   # RADIAL
        4:  (8,  False),  # OPENCV
        5:  (8,  False),  # OPENCV_FISHEYE
        6:  (12, False),  # FULL_OPENCV
        7:  (5,  False),  # FOV
        8:  (4,  True),   # SIMPLE_RADIAL_FISHEYE
        9:  (5,  True),   # RADIAL_FISHEYE
        10: (12, False),  # THIN_PRISM_FISHEYE
    }
    with open(cameras_bin_path, 'rb') as f:
        data = bytearray(f.read())
    offset = 0
    num_cameras = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    for _ in range(num_cameras):
        offset += 4  # camera_id: uint32
        model_id = struct.unpack_from('<i', data, offset)[0]
        offset += 4  # model_id: int32
        offset += 8  # width: uint64
        offset += 8  # height: uint64
        if model_id not in MODEL_PARAMS:
            raise ValueError(f"Unknown COLMAP camera model_id: {model_id}")
        num_params, single_f = MODEL_PARAMS[model_id]
        struct.pack_into('<d', data, offset, fl_px)       # fx or f
        if not single_f:
            struct.pack_into('<d', data, offset + 8, fl_px)  # fy
        offset += num_params * 8
    with open(cameras_bin_path, 'wb') as f:
        f.write(data)


def _linear_to_srgb_u8(img, acescg=False):
    """Convert a linear float EXR frame (BGR or BGRA) to an 8-bit sRGB BGR image.

    Takes the first three channels. When ``acescg`` is set, the linear values are
    first converted from ACEScg (AP1) to linear Rec.709 primaries. The result is
    then clipped to [0, 1] (highlights are clipped, not tone-mapped), passed
    through the sRGB transfer function, and quantised to uint8.
    """
    c = img[..., :3].astype(np.float32)  # BGR, linear
    if acescg:
        # Matrix is in RGB order; flip channels around the matmul to stay in BGR.
        c = (c[..., ::-1] @ ACEScg_TO_SRGB.T)[..., ::-1]
    c = np.clip(c, 0.0, 1.0)
    srgb = np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _list_frames(d, exts):
    """Return sorted paths of files directly in ``d`` whose extension is in ``exts``.

    Extension matching is case-insensitive so camera/export outputs such as
    ``.JPG`` or ``.EXR`` are picked up on case-sensitive filesystems too.
    """
    return sorted(
        os.path.join(d, name)
        for name in os.listdir(d)
        if os.path.isfile(os.path.join(d, name))
        and os.path.splitext(name)[1].lower() in exts
    )


def extract_exr_sequence(exr_dir, img_dir, scale=1.0, acescg=False):
    """Convert a directory of linear .exr frames into frame_%06d.jpg in img_dir.

    Frames are sorted by filename (zero-padded sequences sort correctly), then
    each is read as linear float, converted to sRGB (ACEScg-aware when
    ``acescg`` is set), optionally downscaled, and written with a contiguous
    1-based index so downstream naming matches the FFmpeg path. Returns the
    number of JPGs written.
    """
    exr_files = _list_frames(exr_dir, {".exr"})
    if not exr_files:
        return 0

    written = 0
    for src in tqdm(exr_files, desc="        Converting EXR → JPG", unit="frame"):
        img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"        [WARN] Could not read EXR: {os.path.basename(src)} — skipping.")
            continue

        out = _linear_to_srgb_u8(img, acescg=acescg)
        if scale != 1.0:
            out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        written += 1
        save_path = os.path.join(img_dir, f"frame_{written:06d}.jpg")
        cv2.imwrite(save_path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return written


def extract_jpg_sequence(jpg_dir, img_dir, scale=1.0):
    """Copy/scale a directory of .jpg/.jpeg frames into frame_%06d.jpg in img_dir.

    Frames are sorted by filename and renumbered with a contiguous 1-based index
    so downstream naming matches the FFmpeg path. With scale == 1.0 the originals
    are copied byte-for-byte (no recompression); otherwise each frame is resized.
    No colour conversion is applied. Returns the number of frames written.
    """
    jpg_files = _list_frames(jpg_dir, {".jpg", ".jpeg"})
    if not jpg_files:
        return 0

    written = 0
    for src in tqdm(jpg_files, desc="        Preparing JPG sequence", unit="frame"):
        written += 1
        save_path = os.path.join(img_dir, f"frame_{written:06d}.jpg")
        if scale == 1.0:
            shutil.copyfile(src, save_path)
        else:
            img = cv2.imread(src)
            if img is None:
                print(f"        [WARN] Could not read JPG: {os.path.basename(src)} — skipping.")
                written -= 1
                continue
            out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(save_path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return written


def _sequence_kind(d):
    """Return the image-sequence kind held directly in directory ``d``.

    "exr" if it contains *.exr, "jpg" if it contains *.jpg/*.jpeg, else None
    (case-insensitive). EXR takes precedence if (unusually) both are present.
    """
    if _list_frames(d, {".exr"}):
        return "exr"
    if _list_frames(d, {".jpg", ".jpeg"}):
        return "jpg"
    return None


def process_video(source_path, scenes_dir, idx, total, overlap=12, scale=1.0, mask_path=None, multi_cams=False, acescg=False, lut_path=None, camera_model=None, loop=False, loop_period=5, loop_num_images=50, vocab_tree_path=None, extra_fe=None, extra_sm=None, extra_ma=None, focal_length_mm=None, sensor_width_mm=36.0):
    # An image sequence (EXR or JPG) arrives as a directory; a video as a file.
    seq_kind = _sequence_kind(source_path) if os.path.isdir(source_path) else None
    if seq_kind:
        base_name = os.path.basename(os.path.normpath(source_path))
        ext = ""
    else:
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        ext = os.path.splitext(source_path)[1]

    print(f"\n[{idx}/{total}] === Processing \"{base_name}{ext}\" ===")

    # Directory layout
    scene_path = os.path.join(scenes_dir, base_name)
    img_dir = os.path.join(scene_path, "images")
    sparse_dir = os.path.join(scene_path, "sparse")
    database_path = os.path.join(scene_path, "database.db")

    # Skip if already reconstructed. A successful run produces sparse/0/cameras.bin;
    # a bare scene folder without that marker means a previous run failed midway.
    completion_marker = os.path.join(sparse_dir, "0", "cameras.bin")
    if os.path.exists(completion_marker):
        print(f"        • Skipping \"{base_name}\" – already reconstructed.")
        return
    if os.path.exists(scene_path):
        print(f"        [WARN] Skipping \"{base_name}\" – folder exists but reconstruction is incomplete.")
        print(f"               Delete \"{scene_path}\" to retry.")
        return

    # Clean slate
    try:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
    except OSError as e:
        print(f"        [ERROR] Could not create directories: {e}")
        return

    # --- Mask Detection & Preparation Logic ---
    final_mask_path = None
    
    # 1. Try deriving from --mask argument
    if mask_path:
        # Case A: mask_path is a root containing <basename>_mask
        candidate = os.path.join(mask_path, f"{base_name}_mask")
        if os.path.isdir(candidate):
            final_mask_path = candidate
        # Case B: mask_path itself is the directory (fallback/direct mode)
        elif os.path.isdir(mask_path):
             final_mask_path = mask_path

    # 2. Try auto-detection in video directory
    if not final_mask_path:
        # Check sibling directory (e.g., video_dir/basename_mask)
        candidate = os.path.join(os.path.dirname(source_path), f"{base_name}_mask")
        if os.path.isdir(candidate):
            final_mask_path = candidate

    # 3. Format check and fix
    if final_mask_path:
        print(f"        • Mask directory: {final_mask_path}")
        # Check for *.jpg.png
        has_jpg_png = glob.glob(os.path.join(final_mask_path, "*.jpg.png"))
        
        # If no .jpg.png found, try to rename .png files
        if not has_jpg_png:
            pngs = glob.glob(os.path.join(final_mask_path, "*.png"))
            if pngs:
                print(f"        • Formatting mask filenames (adding .jpg extension)...")
                renamed_count = 0
                for p in pngs:
                    if p.lower().endswith(".jpg.png"): 
                        continue
                    
                    # Rename frame_XXXXX.png -> frame_XXXXX.jpg.png
                    new_name = p[:-4] + ".jpg.png"
                    try:
                        os.rename(p, new_name)
                        renamed_count += 1
                    except OSError as e:
                        print(f"          [WARN] Failed to rename {os.path.basename(p)}: {e}")
                print(f"          -> Renamed {renamed_count} files.")
            else:
                # Folder exists but no PNGs found?
                print(f"          [WARN] Mask directory exists but contains no .png files. Ignoring.")
                final_mask_path = None

    # 1) Produce frame_%06d.jpg in img_dir — from an EXR sequence, a JPG
    #    sequence (both cv2), or a video (FFmpeg).
    if seq_kind == "exr":
        print("        [1/4] Converting EXR sequence → JPG ...")
        if acescg:
            print("        • ACEScg (AP1) → sRGB colour conversion enabled.")
        if lut_path:
            print("        [WARN] --lut is FFmpeg-only and ignored for EXR input.")
        if extract_exr_sequence(source_path, img_dir, scale=scale, acescg=acescg) == 0:
            print(f"        × No EXR frames converted – skipping \"{base_name}\".")
            return
    elif seq_kind == "jpg":
        print("        [1/4] Preparing JPG sequence ...")
        if acescg or lut_path:
            print("        [WARN] --acescg / --lut do not apply to JPG sequences and are ignored.")
        if extract_jpg_sequence(source_path, img_dir, scale=scale) == 0:
            print(f"        × No JPG frames prepared – skipping \"{base_name}\".")
            return
    else:
        print("        [1/4] Extracting frames ...")
        frame_pattern = os.path.join(img_dir, "frame_%06d.jpg")
        cmd_ffmpeg = [
            FFMPEG, "-loglevel", "error", "-stats", "-i", source_path,
            "-qscale:v", "2"
        ]

        # Build video filters
        filters = []

        # ACEScg to sRGB conversion (Generic transform using zscale)
        if acescg:
            # tin=linear (Linear input), t=iec61966-2-1 (sRGB EOTF output)
            # pin=bt2020 (ACEScg is AP1, bt2020 is closest standard primary in zscale)
            # p=bt709 (sRGB/Rec709 primaries)
            filters.append("zscale=tin=linear:t=iec61966-2-1:pin=bt2020:p=bt709:min=bt2020nc:m=bt709")

        # Apply LUT if provided
        if lut_path:
            # Use lut3d filter for .cube files
            safe_lut_path = lut_path.replace("\\", "/") # FFmpeg filters prefer forward slashes
            filters.append(f"lut3d='{safe_lut_path}'")

        if scale != 1.0:
            filters.append(f"scale=iw*{scale}:ih*{scale}")

        if filters:
            cmd_ffmpeg.extend(["-vf", ",".join(filters)])

        cmd_ffmpeg.append(frame_pattern)

        if not run_command(cmd_ffmpeg, f"        × FFmpeg failed – skipping \"{base_name}\"."):
            return

    # Check if frames were extracted
    all_images = glob.glob(os.path.join(img_dir, "*.jpg"))
    if not all_images:
        print(f"        × No frames extracted – skipping \"{base_name}\".")
        return

    # Compute pixel focal length (used for EXIF, camera_params, and post-BA patching)
    fl_px = None
    real_w = real_h = None
    if focal_length_mm:
        first_img = cv2.imread(all_images[0])
        if first_img is not None:
            real_h, real_w = first_img.shape[:2]
            fl_px = (focal_length_mm / sensor_width_mm) * real_w
        else:
            print(f"        [WARN] Could not read first frame to compute focal length in pixels.")

    # Write focal length to EXIF of all extracted frames
    if focal_length_mm:
        fl_35mm = round(focal_length_mm * 36.0 / sensor_width_mm)
        fl_rational = (round(focal_length_mm * 100), 100)
        exif_dict = {"Exif": {
            piexif.ExifIFD.FocalLength: fl_rational,
            piexif.ExifIFD.FocalLengthIn35mmFilm: fl_35mm,
        }}
        exif_bytes = piexif.dump(exif_dict)
        for img_path in tqdm(all_images, desc="        Writing EXIF focal length", unit="frame"):
            try:
                piexif.insert(exif_bytes, img_path)
            except Exception as e:
                print(f"        [WARN] Could not write EXIF to {os.path.basename(img_path)}: {e}")

    # Inject focal length as camera_params if specified
    if fl_px is not None and not (extra_fe and "ImageReader.camera_params" in extra_fe):
        cx = real_w / 2.0
        cy = real_h / 2.0

        model = camera_model or "OPENCV"
        model_upper = model.upper()
        # COLMAP camera_params layouts:
        #   SIMPLE_PINHOLE         (f, cx, cy)
        #   PINHOLE                (fx, fy, cx, cy)
        #   SIMPLE_RADIAL          (f, cx, cy, k)
        #   RADIAL                 (f, cx, cy, k1, k2)
        #   SIMPLE_RADIAL_FISHEYE  (f, cx, cy, k)
        #   OPENCV / OPENCV_FISHEYE and friends — 8 params (fx, fy, cx, cy, d0..d3)
        if model_upper == "SIMPLE_PINHOLE":
            params_str = f"{fl_px},{cx},{cy}"
        elif model_upper == "PINHOLE":
            params_str = f"{fl_px},{fl_px},{cx},{cy}"
        elif model_upper == "SIMPLE_RADIAL":
            params_str = f"{fl_px},{cx},{cy},0"
        elif model_upper == "RADIAL":
            params_str = f"{fl_px},{cx},{cy},0,0"
        elif model_upper == "SIMPLE_RADIAL_FISHEYE":
            params_str = f"{fl_px},{cx},{cy},0"
        else:
            # OPENCV, OPENCV_FISHEYE, and others default to 8-param format
            params_str = f"{fl_px},{fl_px},{cx},{cy},0,0,0,0"

        print(f"        • Focal length: {focal_length_mm}mm / {sensor_width_mm}mm sensor → {fl_px:.1f}px  (camera_params: {params_str})")

        if not camera_model:
            camera_model = "OPENCV"

        if extra_fe is None:
            extra_fe = {}
        extra_fe["ImageReader.camera_params"] = params_str

        # Attempt to prevent bundle adjustment from refining the focal length
        if extra_ma is None:
            extra_ma = {}
        extra_ma.setdefault("GlobalMapper.ba_refine_focal_length", "0")

    # Optional: Check mask count consistency
    if final_mask_path:
        all_masks = glob.glob(os.path.join(final_mask_path, "*.jpg.png"))
        if len(all_images) != len(all_masks):
            print(f"        [WARN] Frame count mismatch! Images: {len(all_images)}, Masks: {len(all_masks)}")
            print(f"               COLMAP will only apply masks to matching filenames.")

    # 2) Feature extraction (COLMAP)
    print("        [2/4] COLMAP feature_extractor ...")
    cmd_colmap_fe = [
        COLMAP, "feature_extractor",
        "--database_path", database_path,
        "--image_path", img_dir,
    ]

    if camera_model:
        cmd_colmap_fe.extend(["--ImageReader.camera_model", camera_model])

    cmd_colmap_fe.extend(["--FeatureExtraction.use_gpu", "1"])
    
    if multi_cams:
        cmd_colmap_fe.extend(["--ImageReader.single_camera_per_folder", "1"])
    else:
        cmd_colmap_fe.extend(["--ImageReader.single_camera", "1"])

    if final_mask_path:
        cmd_colmap_fe.extend(["--ImageReader.mask_path", final_mask_path])

    # Inject extra feature extraction args
    if extra_fe:
        for k, v in extra_fe.items():
            cmd_colmap_fe.extend([f"--{k}", str(v)])

    if not run_command(cmd_colmap_fe, f"        × feature_extractor failed – skipping \"{base_name}\"."):
        return

    # 3) Sequential matching (COLMAP)
    print("        [3/4] COLMAP sequential_matcher ...")
    cmd_colmap_sm = [
        COLMAP, "sequential_matcher",
        "--database_path", database_path,
        "--SequentialMatching.overlap", str(overlap)
    ]
    if loop:
        cmd_colmap_sm.extend([
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.loop_detection_period", str(loop_period),
            "--SequentialMatching.loop_detection_num_images", str(loop_num_images)
        ])
        if vocab_tree_path:
            cmd_colmap_sm.extend(["--SequentialMatching.vocab_tree_path", vocab_tree_path])
    
    # Inject extra sequential matcher args
    if extra_sm:
        for k, v in extra_sm.items():
            cmd_colmap_sm.extend([f"--{k}", str(v)])

    if not run_command(cmd_colmap_sm, f"        × sequential_matcher failed – skipping \"{base_name}\"."):
        return

    # 4) Sparse reconstruction
    print("        [4/4] COLMAP global mapper ...")
    cmd_mapper = [
        COLMAP, "global_mapper",
        "--database_path", database_path,
        "--image_path", img_dir,
        "--output_path", sparse_dir
    ]
    fail_msg = f"        × colmap global_mapper failed – skipping \"{base_name}\"."

    # Inject extra mapper args
    if extra_ma:
        for k, v in extra_ma.items():
            cmd_mapper.extend([f"--{k}", str(v)])

    if not run_command(cmd_mapper, fail_msg):
        return

    # Re-run bundle adjustment with focal length fixed to the user-specified value.
    # The mapper's BA may still refine the focal length despite the flag, so we:
    # 1. Patch cameras.bin to reset focal length to fl_px
    # 2. Re-run colmap bundle_adjuster with refine_focal_length=0
    # This ensures intrinsics and extrinsics remain consistent.
    if fl_px is not None:
        cameras_bin = os.path.join(sparse_dir, "0", "cameras.bin")
        if os.path.exists(cameras_bin):
            try:
                _patch_cameras_bin_focal_length(cameras_bin, fl_px)
                print(f"        • Patched cameras.bin → focal length reset to {fl_px:.1f}px")
            except Exception as e:
                print(f"        [WARN] Could not patch cameras.bin: {e}")
            else:
                sparse_0_dir_ba = os.path.join(sparse_dir, "0")
                cmd_ba = [
                    COLMAP, "bundle_adjuster",
                    "--input_path", sparse_0_dir_ba,
                    "--output_path", sparse_0_dir_ba,
                    "--BundleAdjustment.refine_focal_length", "0",
                ]
                if not run_command(cmd_ba, "        [WARN] bundle_adjuster (fixed focal) failed — continuing without re-BA"):
                    pass  # Non-fatal: TXT export still proceeds with patched cameras.bin
        else:
            print(f"        [WARN] cameras.bin not found at {cameras_bin} — skipping focal length patch.")

    # Export TXT inside the model folder
    # Keep TXT next to BIN so Blender can import from sparse\0 directly.
    sparse_0_dir = os.path.join(sparse_dir, "0")
    if os.path.exists(sparse_0_dir):
        cmd_convert_1 = [
            COLMAP, "model_converter",
            "--input_path", sparse_0_dir,
            "--output_path", sparse_0_dir,
            "--output_type", "TXT"
        ]
        run_command(cmd_convert_1, "        [WARN] Failed to export TXT to sparse/0", quiet=True)

        # Export TXT to parent sparse\ (for Blender auto-detect)
        cmd_convert_2 = [
            COLMAP, "model_converter",
            "--input_path", sparse_0_dir,
            "--output_path", sparse_dir,
            "--output_type", "TXT"
        ]
        run_command(cmd_convert_2, "        [WARN] Failed to export TXT to sparse/", quiet=True)

    print(f"        ✓ Finished \"{base_name}\"  ({idx}/{total})")

def main():
    parser = argparse.ArgumentParser(description="Batch script for automated photogrammetry tracking workflow.")
    parser.add_argument("videos_dir", help="Directory containing input videos")
    parser.add_argument("scenes_dir", help="Directory to output scenes")
    parser.add_argument("--overlap", type=int, default=12, help="Sequential matching overlap (default: 12)")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scaling factor (default: 1.0)")
    parser.add_argument("--mask", help="Path to mask directory (optional)")
    parser.add_argument("--multi-cams", action="store_true", help="Allow processing multiple videos with different camera settings")
    parser.add_argument("--acescg", action="store_true", help="Convert input ACEScg colorspace to sRGB")
    parser.add_argument("--lut", help="Path to .cube LUT file for color conversion (optional)")
    parser.add_argument("--camera_model", default="SIMPLE_RADIAL", help="Specify COLMAP camera model (e.g., OPENCV, PINHOLE, SIMPLE_RADIAL). Default: SIMPLE_RADIAL")
    parser.add_argument("--loop", action="store_true", help="Enable COLMAP loop detection in sequential matching")
    parser.add_argument("--loop_period", type=int, default=5, help="COLMAP loop detection period (default: 5)")
    parser.add_argument("--loop_num_images", type=int, default=50, help="COLMAP loop detection number of images (default: 50)")
    parser.add_argument("--vocab_tree_path", default="vocab_tree_faiss_flickr100K_words32K.bin", help="Path to vocabulary tree for loop detection (default: vocab_tree_faiss_flickr100K_words32K.bin)")
    parser.add_argument("--extra_fe", help="Extra arguments for feature extraction (JSON string or path to .json file)")
    parser.add_argument("--extra_sm", help="Extra arguments for sequential matching (JSON string or path to .json file)")
    parser.add_argument("--extra_ma", help="Extra arguments for mapping (JSON string or path to .json file)")
    parser.add_argument("--focal_length_mm", type=float, default=None, help="Lens focal length in mm (e.g. 24). Combined with --sensor_width_mm to set COLMAP camera_params.")
    parser.add_argument("--sensor_width_mm", type=float, default=36.0, help="Sensor width in mm (default: 36.0 full-frame). Common values: ARRI LF=36.7, Super35=24.89, MFT=17.3")
    
    # If no arguments provided, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    
    videos_dir = os.path.abspath(args.videos_dir)
    scenes_dir = os.path.abspath(args.scenes_dir)
    mask_path = os.path.abspath(args.mask) if args.mask else None
    lut_path = os.path.abspath(args.lut) if args.lut else None

    # Ensure required folders exist
    if not os.path.isdir(videos_dir):
        print(f"[ERROR] Input folder \"{videos_dir}\" missing.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    try:
        os.makedirs(scenes_dir, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] Could not create output folder \"{scenes_dir}\": {e}")
        input("Press Enter to exit...")
        sys.exit(1)

    # Discover inputs:
    #   - top-level video files
    #   - subfolders that hold an image sequence — EXR or JPG (one subfolder = one scene)
    #   - or, if the input directory itself holds loose sequence frames, the whole
    #     directory is treated as a single sequence (scene = its own name)
    # "*_mask" folders are skipped — they hold PNG masks, not source frames.
    entries = sorted(os.listdir(videos_dir))
    video_files = [
        os.path.join(videos_dir, f) for f in entries
        if os.path.isfile(os.path.join(videos_dir, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ]
    seq_dirs = [
        os.path.join(videos_dir, f) for f in entries
        if os.path.isdir(os.path.join(videos_dir, f))
        and not f.endswith("_mask")
        and _sequence_kind(os.path.join(videos_dir, f))
    ]
    # Loose sequence frames directly under the input dir => the dir is one sequence.
    if _sequence_kind(videos_dir):
        if seq_dirs:
            print("[WARN] Found loose sequence frames AND sequence subfolders in the "
                  "input directory. Both will be processed as separate scenes — move "
                  "the loose frames into their own subfolder if that's not intended.")
        seq_dirs.insert(0, videos_dir)

    sources = video_files + seq_dirs
    total = len(sources)

    if total == 0:
        print(f"[INFO] No video files or image sequences found in \"{videos_dir}\".")
        input("Press Enter to exit...")
        sys.exit(0)

    # Parse extra args if provided (supports JSON string or file path)
    def parse_extra(extra_input):
        if not extra_input: return None
        
        # 1. Try treating it as a file path
        if os.path.isfile(extra_input):
            try:
                with open(extra_input, 'r') as f:
                    print(f"        • Loading extra arguments from: {extra_input}")
                    return json.load(f)
            except Exception as e:
                print(f"        [WARN] Failed to read JSON file {extra_input}: {e}")
                return None

        # 2. Try parsing as a raw JSON string
        try:
            return json.loads(extra_input)
        except json.JSONDecodeError:
            # If it's not JSON and not a file, print a helpful warning
            if extra_input.startswith('{') or extra_input.startswith('['):
                print(f"        [WARN] Failed to parse extra arguments JSON string: {extra_input}")
            else:
                print(f"        [WARN] Extra argument is neither a valid file path nor a valid JSON string: {extra_input}")
            return None

    extra_fe = parse_extra(args.extra_fe)
    extra_sm = parse_extra(args.extra_sm)
    extra_ma = parse_extra(args.extra_ma)

    for idx, source_path in enumerate(sources, 1):
        process_video(
            source_path,
            scenes_dir,
            idx, 
            total, 
            overlap=args.overlap, 
            scale=args.scale, 
            mask_path=mask_path, 
            multi_cams=args.multi_cams,
            acescg=args.acescg,
            lut_path=lut_path,

            camera_model=args.camera_model,
            loop=args.loop,
            loop_period=args.loop_period,
            loop_num_images=args.loop_num_images,
            vocab_tree_path=args.vocab_tree_path,
            extra_fe=extra_fe,
            extra_sm=extra_sm,
            extra_ma=extra_ma,
            focal_length_mm=args.focal_length_mm,
            sensor_width_mm=args.sensor_width_mm
        )

    print("--------------------------------------------------------------")
    print(f" All jobs finished – results are in \"{scenes_dir}\".")
    print("--------------------------------------------------------------")

if __name__ == "__main__":
    main()
