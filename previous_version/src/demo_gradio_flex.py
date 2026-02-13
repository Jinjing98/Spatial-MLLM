# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
import tempfile

# sys.path.append("vggt/")
sys.path.append("models/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
# JJ : use from_pretrained to respect HF_HOME
model = VGGT.from_pretrained("facebook/VGGT-1B")


model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# JJ : resolve save_root (default to CWD if empty, raise if not exist)
# -------------------------------------------------------------------------
def resolve_save_root(save_root):
    """
    Resolve save_root directory. Returns the resolved path or None.
    If empty/None, returns None (no local saving).
    Raises ValueError if the provided path does not exist.
    """
    if not save_root or save_root.strip() == "":
        return None
    save_root = save_root.strip()
    if not os.path.isdir(save_root):
        raise ValueError(f"Save root directory does not exist: '{save_root}'. Please create it first.")
    return save_root


# -------------------------------------------------------------------------
# JJ : validate sampling params (total_frames vs sample_fps)
# -------------------------------------------------------------------------
def validate_sampling_params(total_frames, sample_fps):
    """
    Validate and resolve sampling params. Returns (total_frames_int_or_None, sample_fps_float_or_None).
    Raises ValueError if both are set or both are unset.
    """
    # Treat 0 or None as "not set"
    tf = int(total_frames) if (total_frames is not None and total_frames > 0) else None
    sf = float(sample_fps) if (sample_fps is not None and sample_fps > 0) else None

    if tf is not None and sf is not None:
        raise ValueError("Cannot set both Total Frames and Sample FPS. Please clear one of them (set to 0 or empty).")
    if tf is None and sf is None:
        raise ValueError("Must set either Total Frames or Sample FPS (at least one must be > 0).")
    return tf, sf


# -------------------------------------------------------------------------
# JJ : load existing results directory
# -------------------------------------------------------------------------
def load_existing_results(existing_dir):
    """
    Load an existing results directory that contains:
    - images/ subfolder (optional)
    - predictions.npz (optional, but needed for reconstruction)
    - *.glb files (optional, for direct viewing)
    
    At least one of: images/, predictions.npz, or *.glb must exist.
    
    Returns: (target_dir, image_paths, log_message, frame_filter_choices, glb_files)
    """
    if not existing_dir or existing_dir.strip() == "":
        return None, None, "Please provide a directory path.", ["All"], []
    
    existing_dir = existing_dir.strip()
    
    # Convert to absolute path if relative
    if not os.path.isabs(existing_dir):
        existing_dir = os.path.abspath(existing_dir)
        print(f"[DEBUG] Converted to absolute path: {existing_dir}")
    
    # Check if directory exists
    if not os.path.isdir(existing_dir):
        return None, None, f"Directory does not exist: '{existing_dir}'", ["All"], []
    
    # Check for existing GLB files - use absolute paths
    glb_pattern = os.path.join(existing_dir, "*.glb")
    print(f"[DEBUG] Searching for GLB files with pattern: {glb_pattern}")
    glb_files = glob.glob(glb_pattern)
    glb_files = [os.path.abspath(f) for f in glb_files]  # Ensure absolute paths
    glb_files = sorted(glb_files, key=os.path.getmtime, reverse=True)  # Sort by modification time, newest first
    print(f"[DEBUG] Found {len(glb_files)} GLB files")
    
    # Check if predictions.npz exists
    predictions_path = os.path.join(existing_dir, "predictions.npz")
    has_predictions = os.path.exists(predictions_path)
    print(f"[DEBUG] predictions.npz exists: {has_predictions}")
    
    # Check if images/ subfolder exists (optional)
    images_dir = os.path.join(existing_dir, "images")
    image_paths = []
    frame_filter_choices = ["All"]
    
    if os.path.isdir(images_dir):
        print(f"[DEBUG] images/ subfolder found")
        # Load image list
        image_paths = glob.glob(os.path.join(images_dir, "*"))
        image_paths = sorted(image_paths)
        print(f"[DEBUG] Found {len(image_paths)} images")
        
        # Build frame_filter choices
        if len(image_paths) > 0:
            all_files = [os.path.basename(p) for p in image_paths]
            all_files_labeled = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
            frame_filter_choices = ["All"] + all_files_labeled
    else:
        print(f"[DEBUG] images/ subfolder not found (optional)")
    
    # Validate: at least one of images/, predictions.npz, or GLB must exist
    if len(image_paths) == 0 and not has_predictions and len(glb_files) == 0:
        return None, None, f"Directory '{existing_dir}' contains no images, predictions.npz, or GLB files.", ["All"], []
    
    # Build log message
    log_msg = f"Loaded directory: {os.path.basename(existing_dir)}."
    if len(image_paths) > 0:
        log_msg += f" {len(image_paths)} images found."
    else:
        log_msg += " No images/ directory found."
    if has_predictions:
        log_msg += " Predictions found."
    if len(glb_files) > 0:
        log_msg += f" {len(glb_files)} GLB file(s) found - select one from the dropdown below."
        if len(image_paths) == 0:
            log_msg += " Note: Cannot regenerate GLB with different parameters without images/."
    
    return existing_dir, image_paths, log_msg, frame_filter_choices, glb_files


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # JJ : wrap inference in try/finally to ensure GPU cleanup on OOM
    predictions = None
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        # Convert pose encoding to extrinsic and intrinsic matrices
        print("Converting pose encoding to extrinsic and intrinsic matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
        predictions['pose_enc_list'] = None # remove pose_enc_list

        # Generate world points from depth map
        print("Computing world points from depth map...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points

        return predictions
    except torch.cuda.OutOfMemoryError:
        print("ERROR: CUDA OOM during inference. Releasing GPU memory...")
        if predictions is not None:
            del predictions
        raise
    finally:
        del images
        gc.collect()
        torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images, total_frames, sample_fps, save_root=""):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # JJ : validate sampling params
    tf, sf = validate_sampling_params(total_frames, sample_fps)

    # JJ : resolve save_root (None means use temp dir, no persistent saving)
    root = resolve_save_root(save_root)

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if root is not None:
        target_dir = os.path.join(root, f"input_images_{timestamp}")
    else:
        target_dir = os.path.join(tempfile.mkdtemp(), f"input_images_{timestamp}")
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

        # JJ : compute frame_interval based on UI params
        if tf is not None:
            frame_interval = max(1, total_frame_count // tf)
        else:
            frame_interval = max(1, int(fps * sf))

        print(f"Video: {total_frame_count} total frames, fps={fps:.1f}, frame_interval={frame_interval}")

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images, total_frames, sample_fps, save_root=""):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None, None
    try:
        target_dir, image_paths = handle_uploads(input_video, input_images, total_frames, sample_fps, save_root)
    except ValueError as e:
        return None, None, None, str(e)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    # JJ : wrap reconstruction in try/except to handle OOM gracefully
    predictions = None
    try:
        print("Running run_model...")
        with torch.no_grad():
            predictions = run_model(target_dir, model)

        # Save predictions
        prediction_save_path = os.path.join(target_dir, "predictions.npz")
        np.savez(prediction_save_path, **predictions)

        # Handle None frame_filter
        if frame_filter is None:
            frame_filter = "All"

        # Build a GLB file name
        glbfile = os.path.join(
            target_dir,
            f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
        )

        # Convert predictions to GLB
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

        end_time = time.time()
        print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
        log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

        return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)
    except torch.cuda.OutOfMemoryError:
        log_msg = "CUDA Out of Memory! Try reducing the number of frames (lower Total Frames or increase Sample FPS interval)."
        print(f"ERROR: {log_msg}")
        return None, log_msg, gr.Dropdown(choices=frame_filter_choices, value="All", interactive=True)
    finally:
        if predictions is not None:
            del predictions
        gc.collect()
        torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    # JJ : Check if images/ directory exists (required for generating new GLB files)
    images_dir = os.path.join(target_dir, "images")
    has_images = os.path.isdir(images_dir)

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        # JJ : If no images/ directory, cannot generate new GLB files
        if not has_images:
            # Find any existing GLB file in the directory as fallback
            existing_glbs = glob.glob(os.path.join(target_dir, "*.glb"))
            if len(existing_glbs) > 0:
                fallback_glb = existing_glbs[0]
                return fallback_glb, "Cannot generate new GLB (no images/ directory). Showing existing GLB. Please use the GLB selector dropdown to choose different files."
            else:
                return None, "Cannot generate GLB: images/ directory not found."
        
        # Generate new GLB if images/ exists
        key_list = [
            "pose_enc",
            "depth",
            "depth_conf",
            "world_points",
            "world_points_conf",
            "images",
            "extrinsic",
            "intrinsic",
            "world_points_from_depth",
        ]
        loaded = np.load(predictions_path)
        predictions = {key: np.array(loaded[key]) for key in key_list}
        
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# JJ : handler for loading existing results
# -------------------------------------------------------------------------
def on_load_existing_results(existing_dir):
    """
    Handler for the Load button. Clears upload fields and loads existing results.
    Populates the GLB file selector if GLB files are found.
    Returns: (video_clear, images_clear, target_dir, gallery, log_msg, frame_filter_dropdown, glb_selector_dropdown)
    """
    print(f"[DEBUG] on_load_existing_results called with: {existing_dir}")
    target_dir, image_paths, log_msg, frame_filter_choices, glb_files = load_existing_results(existing_dir)
    
    # If loading failed
    if target_dir is None:
        print(f"[DEBUG] Loading failed")
        return (
            None, None, None, None, log_msg, 
            gr.Dropdown(choices=["All"], value="All", interactive=True),
            gr.Dropdown(choices=[], value=None, visible=False)
        )
    
    print(f"[DEBUG] Loaded target_dir: {target_dir}")
    print(f"[DEBUG] Found {len(glb_files)} GLB files:")
    for i, gf in enumerate(glb_files):
        print(f"[DEBUG]   GLB {i}: {gf}")
    
    # Build GLB file choices with friendly names
    glb_choices = []
    for glb_path in glb_files:
        basename = os.path.basename(glb_path)
        # Add file size and modification time for context
        size_mb = os.path.getsize(glb_path) / (1024 * 1024)
        mtime = datetime.fromtimestamp(os.path.getmtime(glb_path)).strftime("%Y-%m-%d %H:%M:%S")
        friendly_name = f"{basename} ({size_mb:.1f}MB, {mtime})"
        glb_choices.append((friendly_name, glb_path))  # (display_name, value)
        print(f"[DEBUG]   Choice: '{friendly_name}' -> '{glb_path}'")
    
    # Prepare GLB selector dropdown
    if len(glb_files) > 0:
        default_glb = glb_files[0]
        print(f"[DEBUG] Setting default GLB to: {default_glb}")
        glb_dropdown = gr.Dropdown(
            choices=glb_choices,
            value=default_glb,  # Select the first (newest) GLB by default
            visible=True,
            interactive=True
        )
    else:
        print(f"[DEBUG] No GLB files, hiding dropdown")
        glb_dropdown = gr.Dropdown(choices=[], value=None, visible=False)
    
    # Return: clear video, clear images, set target_dir, update gallery, update log, update frame_filter dropdown, update glb_selector
    return (
        None, None, target_dir, image_paths, log_msg,
        gr.Dropdown(choices=frame_filter_choices, value="All", interactive=True),
        glb_dropdown
    )


# -------------------------------------------------------------------------
# JJ : handler for GLB file selection
# -------------------------------------------------------------------------
def on_glb_selected(glb_path):
    """
    Handler for when user selects a GLB file from the dropdown.
    Simply loads and displays the selected GLB file.
    Returns: (reconstruction_output, log_msg)
    """
    print(f"[DEBUG] on_glb_selected called")
    print(f"[DEBUG] glb_path type: {type(glb_path)}")
    print(f"[DEBUG] glb_path value: {glb_path}")
    
    if not glb_path:
        print(f"[DEBUG] glb_path is None or empty")
        return None, "No GLB file selected."
    
    if not os.path.exists(glb_path):
        print(f"[DEBUG] GLB file does not exist: {glb_path}")
        # Try to resolve relative path
        if not os.path.isabs(glb_path):
            abs_path = os.path.abspath(glb_path)
            print(f"[DEBUG] Trying absolute path: {abs_path}")
            if os.path.exists(abs_path):
                glb_path = abs_path
            else:
                return None, f"Selected GLB file not found: {glb_path}"
        else:
            return None, f"Selected GLB file not found: {glb_path}"
    
    print(f"[DEBUG] Returning GLB file: {glb_path}")
    basename = os.path.basename(glb_path)
    return glb_path, f"Displaying: {basename}"


# -------------------------------------------------------------------------
# Example images
# -------------------------------------------------------------------------

great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # Instead of gr.State, we use a hidden Textbox:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Pointmap Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            # JJ : save root directory (empty = no local saving, must exist if set)
            ui_save_root = gr.Textbox(
                label="Save Root (empty = no local saving, must exist if set)",
                value="", placeholder="/path/to/output/dir", interactive=True,
            )
            
            # JJ : load existing results directory
            gr.Markdown("---")
            gr.Markdown("**Load Existing Results** *(alternative to uploading new data)*")
            with gr.Row():
                ui_existing_dir = gr.Textbox(
                    label="Existing Results Directory",
                    value="", 
                    placeholder="/path/to/existing/results/input_images_XXX",
                    interactive=True,
                    scale=4,
                )
                load_btn = gr.Button("Load", scale=1, variant="secondary")
            
            # JJ : GLB file selector - populated after loading directory
            ui_glb_selector = gr.Dropdown(
                label="Select GLB File to Display",
                choices=[],
                value=None,
                interactive=True,
                visible=False,
            )
            gr.Markdown("---")

            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            # JJ : flexible video sampling controls
            gr.Markdown("**Video Frame Sampling** *(set one, leave the other as 0)*")
            with gr.Row():
                ui_total_frames = gr.Number(
                    label="Total Frames (0 = disabled)",
                    value=0, precision=0, minimum=0, interactive=True,
                )
                ui_sample_fps = gr.Number(
                    label="Sample FPS interval in sec (0 = disabled)",
                    value=1.0, minimum=0, interactive=True,
                )

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery, ui_save_root, ui_existing_dir, ui_glb_selector],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # ---------------------- Examples section ----------------------
    examples = [
        [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True"],
        [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
    ]

    def example_pipeline(
        input_video,
        num_images_str,
        input_images,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example_str,
    ):
        """
        1) Copy example images to new target_dir
        2) Reconstruct
        3) Return model3D + logs + new_dir + updated dropdown + gallery
        We do NOT return is_example. It's just an input.
        """
        # JJ : examples use default sample_fps=1.0
        target_dir, image_paths = handle_uploads(input_video, input_images, total_frames=0, sample_fps=1.0)
        # Always use "All" for frame_filter in examples
        frame_filter = "All"
        glbfile, log_msg, dropdown = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # JJ : Note - when directory has no images/, these will return existing GLB as fallback
    # -------------------------------------------------------------------------
    viz_inputs = [
        target_dir_output,
        conf_thres,
        frame_filter,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example,
    ]
    viz_outputs = [reconstruction_output, log_output]

    conf_thres.change(update_visualization, viz_inputs, viz_outputs)
    frame_filter.change(update_visualization, viz_inputs, viz_outputs)
    mask_black_bg.change(update_visualization, viz_inputs, viz_outputs)
    mask_white_bg.change(update_visualization, viz_inputs, viz_outputs)
    show_cam.change(update_visualization, viz_inputs, viz_outputs)
    mask_sky.change(update_visualization, viz_inputs, viz_outputs)
    prediction_mode.change(update_visualization, viz_inputs, viz_outputs)

    # -------------------------------------------------------------------------
    # JJ : Auto-update gallery whenever user uploads, changes files,
    #       or adjusts sampling params (total_frames / sample_fps)
    # -------------------------------------------------------------------------
    upload_inputs = [input_video, input_images, ui_total_frames, ui_sample_fps, ui_save_root]
    upload_outputs = [reconstruction_output, target_dir_output, image_gallery, log_output]

    input_video.change(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)
    input_images.change(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)
    ui_total_frames.change(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)
    ui_sample_fps.change(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)
    ui_save_root.change(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)

    # -------------------------------------------------------------------------
    # JJ : Load existing results button handler
    # -------------------------------------------------------------------------
    load_btn.click(
        fn=on_load_existing_results,
        inputs=[ui_existing_dir],
        outputs=[input_video, input_images, target_dir_output, image_gallery, log_output, frame_filter, ui_glb_selector]
    ).then(
        fn=on_glb_selected,  # Auto-display the first GLB if available
        inputs=[ui_glb_selector],
        outputs=[reconstruction_output, log_output]
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False" so visualization updates work
    )
    
    # -------------------------------------------------------------------------
    # JJ : GLB file selector change handler
    # -------------------------------------------------------------------------
    ui_glb_selector.change(
        fn=on_glb_selected,
        inputs=[ui_glb_selector],
        outputs=[reconstruction_output, log_output]
    )

    # JJ : bind 0.0.0.0 so it is accessible from outside (e.g. SLURM nodes)
    demo.queue(max_size=20).launch(show_error=True, share=False, server_name="0.0.0.0")
