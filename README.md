# AI Colmap Camera Tracking

Automated pipeline for camera tracking and scene reconstruction using COLMAP and NeRF-compatible formats. Processes video inputs, performs sparse 3D reconstruction, undistorts footage, and exports camera data for use in Houdini or NeRF training.

## Features

- **Automated Workflow:** Batch processes multiple video files.
- **Frame Extraction:** Uses FFmpeg to extract frames from input videos.
- **Feature Extraction & Matching:** Utilises COLMAP for feature extraction and sequential matching.
- **Sparse Reconstruction:** Uses **COLMAP Global Mapper** (requires COLMAP 4.0+).
- **NeRF Conversion:** Converts COLMAP data to `transforms.json` (NeRF format).
- **Undistortion:** Expands the undistorted canvas to preserve all valid pixels while keeping the focal length unchanged. Optionally crops to the original canvas size (`--crop`).
- **Houdini Integration:** Automatically generates a Houdini `.hip` scene with the reconstructed point cloud and animated camera. Correctly handles sensor size, canvas expansion, and principal-point offset.

## Prerequisites

1. **[uv](https://docs.astral.sh/uv/)** — Python package and environment manager.
2. **FFmpeg** — video processing.
3. **[COLMAP 4.0+](https://github.com/colmap/colmap)** — feature extraction, matching, and Global Mapper reconstruction.
4. **Houdini (hython)** — required only for Houdini scene generation.
5. **Vocabulary tree** *(optional)* — required only when using `--loop` for loop detection.
   Download `vocab_tree_faiss_flickr100K_words32K.bin` (≈9 MB) from the COLMAP project
   (<https://demuc.de/colmap/>) and place it in the repo root, or pass its path via
   `--vocab_tree_path`. The file is intentionally not tracked in git.

### Python Dependencies

Dependencies are declared in `pyproject.toml` and managed by `uv`. To create the virtual environment and install all dependencies:

```bash
uv sync
```

*Optional:* For automatic object masking in `colmap2nerf.py`, PyTorch and Detectron2 are required.

## Usage

The main entry point is `run_autotracker.py`.

```bash
uv run python run_autotracker.py <input_videos_dir> <output_dir> [options]
```

## Graphical User Interface (GUI)

A PySide6-based GUI is available for a more user-friendly experience. It wraps the `run_autotracker.py` script and provides a real-time log of the processing steps.

### Launching the GUI

```bash
uv run gui_autotracker.py
```

The GUI allows you to browse for input/output directories, adjust processing scales, select COLMAP camera models, and toggle advanced settings like loop detection or Houdini path configuration.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `input_videos_dir` | — | Directory containing source video files (`.mp4`, `.mov`, …) |
| `output_dir` | — | Directory for all output data |
| `--scale` | `0.5` | Image scaling factor applied before processing |
| `--overlap` | `12` | Sequential matching overlap (number of frames) |
| `--skip-houdini` | off | Skip Houdini `.hip` generation |
| `--hfs` | — | Path to Houdini installation directory (e.g. `C:\Program Files\Side Effects Software\Houdini 20.0.625`). If omitted, `hython` must be in `PATH`. |
| `--multi-cams` | off | Treat each video as a separate camera (useful for multi-device shoots) |
| `--acescg` | off | Convert input from ACEScg to sRGB before processing |
| `--lut` | — | Path to a `.cube` LUT file for colour-space conversion |
| `--mask` | — | Path to a directory containing per-frame masks |
| `--focal_length_mm` | — | Lens focal length in mm (e.g. `24`). Locks COLMAP to this value instead of estimating it. |
| `--sensor_width_mm` | `36.0` | Physical sensor width in mm. Used together with `--focal_length_mm`. Common values: full-frame=36.0, ARRI LF=36.7, Super35=24.89, MFT=17.3 |
| `--crop` | off | Keep original canvas size during undistortion instead of expanding it. Houdini focal length and aperture remain at exact physical values (e.g. 20 mm / 36 mm). |
| `--camera_model` | auto | COLMAP camera model (e.g. `OPENCV`, `PINHOLE`, `SIMPLE_RADIAL`) |
| `--loop` | off | Enable loop detection in sequential matching |
| `--loop_period` | `5` | Loop detection period |
| `--loop_num_images` | `50` | Number of images considered per loop detection pass |
| `--vocab_tree_path` | `vocab_tree_faiss_flickr100K_words32K.bin` | Path to vocabulary tree for loop detection |
| `--extra_fe` | — | Extra COLMAP feature-extraction arguments. Accepts a JSON string or `.json` file path. |
| `--extra_sm` | — | Extra COLMAP sequential-matching arguments. |
| `--extra_ma` | — | Extra COLMAP global-mapper arguments. |

### Specifying Focal Length

When the shooting focal length is known, providing it improves reconstruction accuracy by preventing COLMAP from freely estimating it:

```bash
python run_autotracker.py ./videos ./output --focal_length_mm 24 --sensor_width_mm 36
```

### Extra Arguments Example

Create a `params.json`:
```json
{
    "SiftExtraction.peak_threshold": 0.01,
    "SiftExtraction.max_num_features": 8192
}
```

Then pass it:
```bash
uv run python run_autotracker.py ./in ./out --extra_fe params.json
```

### Masking

The pipeline supports per-frame masks to exclude moving objects or unwanted regions from reconstruction.

**Rules:**
1. **Auto-detection:** For a video `shot01.mp4`, the script looks for a sibling directory named `shot01_mask`.
2. **Custom root:** `--mask <path>` looks for `<video_name>_mask` inside the specified path.
3. **Filename format:** PNG files named `frame_000001.jpg.png`. If `frame_000001.png` is found it is automatically renamed to match COLMAP requirements.

### Example

```bash
uv run python run_autotracker.py ./videos ./output \
    --scale 0.5 \
    --focal_length_mm 20 \
    --sensor_width_mm 36 \
    --hfs "C:/Program Files/Side Effects Software/Houdini 20.0.625"
```

## Batch Processing

`batch_run.py` processes multiple folders within a target directory. Per-folder settings can be defined in a `batch_config.ini` placed in the target directory.

### Usage

```bash
uv run python batch_run.py <target_directory>
```

### Configuration Format (`batch_config.ini`)

If a section name matches a folder name, its settings override the defaults for that folder.

```ini
[global]
scale = 0.5
hfs = C:/Program Files/Side Effects Software/Houdini 20.0.625

[shot_01]
scale = 0.8
camera_model = OPENCV
focal_length_mm = 24
sensor_width_mm = 36

[shot_02]
focal_length_mm = 20
sensor_width_mm = 24.89
acescg = true
```

**Supported INI Keys:**

| Key | Type | Description |
|---|---|---|
| `scale` | float | Image scaling factor |
| `overlap` | int | Sequential matching overlap |
| `camera_model` | string | e.g. `OPENCV`, `PINHOLE` |
| `focal_length_mm` | float | Lens focal length in mm |
| `sensor_width_mm` | float | Physical sensor width in mm |
| `mask` | string | Path to mask directory |
| `lut` | string | Path to `.cube` LUT file |
| `hfs` | string | Path to Houdini installation |
| `crop` | bool | `true` / `false` |
| `multi_cams` | bool | `true` / `false` |
| `acescg` | bool | `true` / `false` |
| `skip_houdini` | bool | `true` / `false` |
| `loop` | bool | `true` / `false` |
| `loop_period` | int | Loop detection period |
| `loop_num_images` | int | Images per loop detection pass |
| `vocab_tree_path` | string | Path to vocabulary tree |

### Advanced Parameter Injection (INI Only)

Pass any COLMAP internal parameter using these prefixes:

- `fe.<Parameter>` — injected into `feature_extractor`
- `sm.<Parameter>` — injected into `sequential_matcher`
- `ma.<Parameter>` — injected into `global_mapper`

**Example:**
```ini
[global]
fe.SiftExtraction.peak_threshold = 0.01
sm.SequentialMatching.min_num_matches = 20
```

## Quick Start / Demo

```bash
run_demo_test.bat
```

Processes `./demo-test/walking-forest` and outputs to `./demo-test/walking-forest-output`. Edit the `.bat` file to point to your Houdini installation if `hython` is not in `PATH`.

## Pipeline Steps

1. **Frame extraction** — FFmpeg extracts frames from each input video.
2. **Feature extraction & matching** — COLMAP `feature_extractor` + `sequential_matcher`.
3. **Sparse reconstruction** — COLMAP `global_mapper` (requires COLMAP 4.0+).
4. **Model export** — Sparse model converted to TXT and PLY formats.
5. **NeRF conversion** — `colmap2nerf.py` generates `transforms.json`.
6. **Undistortion** — `undistortionNerfstudioColmap.py` removes lens distortion.
   - Default: canvas is expanded to include all valid pixels; `sensor_w`/`sensor_h` are recorded so downstream tools can recover the physical focal length.
   - `--crop`: keeps the original canvas size; Houdini focal length and aperture remain at their exact physical values (e.g. 20 mm / 36 mm).
7. **Houdini scene** — `build_houdini_scene.py` imports the point cloud and creates an animated camera with correct focal length, aperture, and principal-point offset.

## Scripts Overview

| Script | Description |
|---|---|
| `run_autotracker.py` | Master script — orchestrates the full pipeline |
| `autotracker.py` | Core photogrammetry: FFmpeg, COLMAP feature extraction, matching, and Global Mapper |
| `colmap2nerf.py` | Converts COLMAP sparse model to `transforms.json` |
| `undistortionNerfstudioColmap.py` | Undistorts images; expands canvas or crops to original size |
| `restore_distortion.py` | Utility to apply or remove lens distortion from rendered images. Supports EXR via `--exr` |
| `build_houdini_scene.py` | Generates a `.hip` file with point cloud and animated camera |
| `batch_run.py` | Batch runner with per-folder INI configuration |

## Output Structure

For each processed video:

```
<output_dir>/<video_name>/
├── images/                  # Extracted frames
├── sparse/                  # COLMAP sparse reconstruction
├── database.db              # COLMAP feature database
├── transforms.json          # Camera poses (NeRF format)
├── points3D.ply             # Point cloud
├── undistort/
│   ├── images_undistorted/  # Undistorted frames
│   └── transforms_undistorted.json
└── <video_name>.hip         # Houdini project file
```

## References

- Inspired by: [Video Link](https://youtu.be/xx85eyN1Xc0?si=icXcANMb06k-v9dE)
- Demo test video: [Pexels — Tranquil Autumn Forest Walkway Path](https://www.pexels.com/video/tranquil-autumn-forest-walkway-path-29142343/)
