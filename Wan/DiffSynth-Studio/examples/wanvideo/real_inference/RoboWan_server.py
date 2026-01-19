"""
RoboWan Server - FastAPI Version for Multi-View Heatmap Inference
Runs on GPU server to receive observations and return actions (position, rotation, gripper)
"""

import numpy as np
import torch
from PIL import Image
import io
import sys
import os
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import traceback
import base64

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Auto-detect root path
def get_root_path():
    """Auto-detect BridgeVLA root directory"""
    possible_paths = [
        "/share/project/lpy/BridgeVLA",
        "/home/lpy/BridgeVLA_dev",
        "/DATA/disk0/lpy/BridgeVLA_dev",
        "/DATA/disk1/lpy_a100_4/BridgeVLA_dev",
        "/DATA/disk1/lpy_a100_1/BridgeVLA_dev",
        "/DATA/disk1/lpy/BridgeVLA_dev"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Cannot find BridgeVLA root directory in any of: {possible_paths}")

ROOT_PATH = get_root_path()
print(f"Using ROOT_PATH: {ROOT_PATH}")

# Add project paths
diffsynth_path = os.path.abspath(os.path.join(ROOT_PATH, "Wan/DiffSynth-Studio"))
model_inference_path = os.path.join(ROOT_PATH, "Wan/DiffSynth-Studio/examples/wanvideo/model_inference")
sys.path.insert(0, diffsynth_path)
sys.path.insert(0, model_inference_path)

# Import the inference class from model_inference
# Old version (VAE decode feature, channel concat):
# from heatmap_inference_TI2V_5B_fused_mv_rot_grip_vae_decode_feature_3zed import (
#     HeatmapInferenceMVRotGrip,
#     get_3d_position_from_pred_heatmap
# )
# New version (View concat with heatmap images):
from heatmap_inference_mv_view_rot_grip import (
    HeatmapInferenceMVViewRotGrip as HeatmapInferenceMVRotGrip,  # Use alias for compatibility
    get_3d_position_from_pred_heatmap
)
# ProjectionInterface will be imported based on use_different_projection flag
import bridgevla.mvt.utils as mvt_utils


def get_projection_interface_class(use_different_projection: bool):
    """
    根据投影模式返回对应的ProjectionInterface类

    Args:
        use_different_projection: 是否使用不同投影模式（每个相机单独投影）

    Returns:
        ProjectionInterface class
    """
    if use_different_projection:
        from diffsynth.trainers.base_multi_view_dataset_with_rot_grip_3cam_different_projection import ProjectionInterface
        print("Using different projection mode (3cam_different_projection)")
    else:
        from diffsynth.trainers.base_multi_view_dataset_with_rot_grip import ProjectionInterface
        print("Using default projection mode")
    return ProjectionInterface


class RoboWanInferenceEngine:
    """Inference engine wrapper using HeatmapInferenceMVRotGrip"""

    def __init__(
        self,
        lora_checkpoint: str,
        rot_grip_checkpoint: str,
        model_base_path: str,
        wan_type: str,
        use_dual_head: bool,
        rotation_resolution: float,
        hidden_dim: int,
        num_rotation_bins: int,
        local_feat_size: int = 5,
        scene_bounds: List[float] = None,
        img_size: int = 256,
        device: str = "cuda",
        use_different_projection: bool = False,
        use_initial_gripper_state: bool = False,
        num_history_frames: int = 1
    ):
        self.device = device
        self.rotation_resolution = rotation_resolution
        self.num_rotation_bins = num_rotation_bins
        self.scene_bounds = scene_bounds
        self.img_size = img_size
        self.use_different_projection = use_different_projection
        self.use_initial_gripper_state = use_initial_gripper_state
        self.num_history_frames = num_history_frames

        print(f"Initializing {wan_type} inference engine...")
        print(f"Num history frames: {num_history_frames}")
        print(f"Use initial gripper state: {use_initial_gripper_state}")

        # Use HeatmapInferenceMVRotGrip for consistent initialization
        # Note: Parameter names updated for HeatmapInferenceMVViewRotGrip compatibility
        self.inference_engine = HeatmapInferenceMVRotGrip(
            model_base_path=model_base_path,
            lora_checkpoint=lora_checkpoint,  # Changed from lora_checkpoint_path
            rot_grip_checkpoint=rot_grip_checkpoint,  # Changed from rot_grip_checkpoint_path
            wan_type=wan_type,
            use_dual_head=use_dual_head,
            use_merged_pointcloud=False,  # Added: not used in server mode
            use_different_projection=use_different_projection,  # Added: match training config
            rotation_resolution=rotation_resolution,
            hidden_dim=hidden_dim,
            num_rotation_bins=num_rotation_bins,
            num_history_frames=num_history_frames,
            local_feat_size=local_feat_size,
            use_initial_gripper_state=use_initial_gripper_state,
            device=device,
            is_full_finetune=False,  # Added: using LoRA checkpoint
            torch_dtype=torch.bfloat16
        )

        # Initialize projection interface for position extraction
        # 根据use_different_projection选择正确的ProjectionInterface
        ProjectionInterface = get_projection_interface_class(use_different_projection)
        self.projection_interface = ProjectionInterface(
            img_size=img_size,
            rend_three_views=True,
            add_depth=False,
            device=device  # 使用与模型相同的设备
        )

        print("✓ Model loaded successfully!")

    def _create_rev_trans(self):
        """Create reverse transformation function from cube coordinates to world coordinates"""
        # Use a dummy point to create the transformation
        # The transformation is determined by scene_bounds only
        dummy_point = np.array([[0.0, 0.0, 0.0]])
        _, rev_trans = mvt_utils.place_pc_in_cube(
            torch.from_numpy(dummy_point),
            scene_bounds=self.scene_bounds,
            with_mean_or_bounds=False
        )
        return rev_trans

    @torch.no_grad()
    def predict(
        self,
        prompt: str,
        input_image: List[Image.Image] = None,
        input_image_rgb: List[Image.Image] = None,
        input_images: List[List[Image.Image]] = None,
        input_images_rgb: List[List[Image.Image]] = None,
        initial_rotation: np.ndarray = None,
        initial_gripper: int = None,
        num_frames: int = 13
    ) -> Dict[str, Any]:
        """
        Run inference to predict position, rotation, and gripper

        Args:
            prompt: Text instruction
            input_image: [单帧模式] List of PIL Images for heatmap input (multi-view)
            input_image_rgb: [单帧模式] List of PIL Images for RGB input (multi-view)
            input_images: [多帧历史模式] List[List[PIL.Image]] - (num_history, num_views)
            input_images_rgb: [多帧历史模式] List[List[PIL.Image]] - (num_history, num_views)
            initial_rotation: Initial rotation in degrees [roll, pitch, yaw]
            initial_gripper: Initial gripper state (0 or 1)
            num_frames: Total number of frames including initial frame (default 13)

        Returns:
            Dictionary containing:
                - position: Predicted positions (num_frames-1, 3) in meters
                - rotation: Predicted rotations (num_frames-1, 3) in degrees
                - gripper: Predicted gripper states (num_frames-1,)
                - video_heatmap: Generated heatmap video
                - video_rgb: Generated RGB video
        """
        # Use HeatmapInferenceMVRotGrip.predict() for consistency
        # 根据模式选择参数：单帧模式或多帧历史模式
        if self.num_history_frames > 1 and input_images is not None:
            # 多帧历史模式
            output = self.inference_engine.predict(
                prompt=prompt,
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                input_images=input_images,
                input_images_rgb=input_images_rgb,
                initial_rotation=initial_rotation,
                initial_gripper=initial_gripper,
                num_frames=num_frames,
                height=self.img_size,
                width=self.img_size,
                num_inference_steps=50,
                cfg_scale=1.0
            )
        else:
            # 单帧模式（向后兼容）
            output = self.inference_engine.predict(
                prompt=prompt,
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                initial_rotation=initial_rotation,
                initial_gripper=initial_gripper,
                num_frames=num_frames,
                height=self.img_size,
                width=self.img_size,
                num_inference_steps=50,
                cfg_scale=1.0
            )

        # Extract position from predicted heatmap
        pred_heatmap = output['video_heatmap']  # List[List[PIL.Image]] - (num_views, T) or (T, num_views)
        rev_trans = self._create_rev_trans()

        pred_position = get_3d_position_from_pred_heatmap(
            pred_heatmap_colormap=pred_heatmap,
            rev_trans=rev_trans,
            projection_interface=self.projection_interface,
            colormap_name='jet'
        )  # (num_frames, 3) in world coordinates

        return {
            'position': pred_position,  # (num_frames-1, 3) or (num_frames, 3) depending on implementation
            'rotation': output['rotation_predictions'],  # (num_frames-1, 3)
            'gripper': output['gripper_predictions'],    # (num_frames-1,)
            'video_heatmap': output['video_heatmap'],
            'video_rgb': output['video_rgb']
        }


class RoboWanServer:
    """FastAPI server for RoboWan inference"""

    def __init__(
        self,
        lora_checkpoint: str,
        rot_grip_checkpoint: str,
        model_base_path: str,
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP",
        use_dual_head: bool = True,
        rotation_resolution: float = 5.0,
        hidden_dim: int = 512,
        num_rotation_bins: int = 72,
        local_feat_size: int = 5,
        num_frames: int = 13,
        scene_bounds: List[float] = None,
        img_size: int = 256,
        device: str = "cuda:0",
        use_different_projection: bool = False,
        use_initial_gripper_state: bool = False,
        num_history_frames: int = 1
    ):
        """
        Initialize RoboWan server

        Args:
            lora_checkpoint: Path to LoRA checkpoint
            rot_grip_checkpoint: Path to rotation/gripper predictor checkpoint
            model_base_path: Path to base model
            wan_type: Model type
            use_dual_head: Whether to use dual head mode
            rotation_resolution: Rotation discretization resolution in degrees
            hidden_dim: Hidden dimension for predictor
            num_rotation_bins: Number of rotation bins
            local_feat_size: Local feature size for rotation/gripper predictor
            num_frames: Total number of frames (including initial frame)
            scene_bounds: Scene bounds [x_min, y_min, z_min, x_max, y_max, z_max]
            img_size: Image size for projection interface
            device: Device for inference
            use_different_projection: Whether to use different projection mode (3cam_different_projection)
            use_initial_gripper_state: Whether to use initial gripper state as input (must match training)
            num_history_frames: Number of history frames to use (1, 2, or 1+4N)
        """
        self.device = device
        self.num_frames = num_frames
        self.num_history_frames = num_history_frames

        # Default scene bounds if not provided
        if scene_bounds is None:
            scene_bounds = [0, -0.7, -0.05, 0.8, 0.7, 0.65]

        print(f"Initializing RoboWan server on {device}...")
        print(f"  Scene bounds: {scene_bounds}")
        print(f"  Default num_frames: {num_frames}")
        print(f"  Num history frames: {num_history_frames}")
        print(f"  Image size: {img_size}")
        print(f"  Use different projection: {use_different_projection}")
        print(f"  Use initial gripper state: {use_initial_gripper_state}")

        # Create inference engine
        self.inference_engine = RoboWanInferenceEngine(
            lora_checkpoint=lora_checkpoint,
            rot_grip_checkpoint=rot_grip_checkpoint,
            model_base_path=model_base_path,
            wan_type=wan_type,
            use_dual_head=use_dual_head,
            rotation_resolution=rotation_resolution,
            hidden_dim=hidden_dim,
            num_rotation_bins=num_rotation_bins,
            local_feat_size=local_feat_size,
            scene_bounds=scene_bounds,
            img_size=img_size,
            device=device,
            use_different_projection=use_different_projection,
            use_initial_gripper_state=use_initial_gripper_state,
            num_history_frames=num_history_frames
        )

        print("✓ Server initialized successfully!")

    def predict_action(
        self,
        prompt: str,
        input_image: List[Image.Image] = None,
        input_image_rgb: List[Image.Image] = None,
        input_images: List[List[Image.Image]] = None,
        input_images_rgb: List[List[Image.Image]] = None,
        initial_rotation: List[float] = None,
        initial_gripper: int = None,
        num_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predict robot actions from observations

        Args:
            prompt: Text instruction
            input_image: [单帧模式] List of PIL Images for heatmap (multi-view)
            input_image_rgb: [单帧模式] List of PIL Images for RGB (multi-view)
            input_images: [多帧历史模式] List[List[PIL.Image]] - (num_history, num_views)
            input_images_rgb: [多帧历史模式] List[List[PIL.Image]] - (num_history, num_views)
            initial_rotation: Initial rotation [roll, pitch, yaw] in degrees
            initial_gripper: Initial gripper state (0 or 1)
            num_frames: Number of frames to predict (uses server default if None)

        Returns:
            Dictionary containing predictions
        """
        # Use server default if not specified
        if num_frames is None:
            num_frames = self.num_frames

        # Convert to numpy if provided
        initial_rotation_np = np.array(initial_rotation, dtype=np.float32) if initial_rotation is not None else None

        # 决定使用单帧模式还是多帧历史模式
        if input_images is not None and input_images_rgb is not None:
            # 多帧历史模式
            print(f"  Using multi-frame history mode: {len(input_images)} history frames")
            output = self.inference_engine.predict(
                prompt=prompt,
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                input_images=input_images,
                input_images_rgb=input_images_rgb,
                initial_rotation=initial_rotation_np,
                initial_gripper=initial_gripper,
                num_frames=num_frames
            )

        else:
            # 单帧模式
            print(f"  Using single-frame mode")
            output = self.inference_engine.predict(
                prompt=prompt,
                input_image=input_image,
                input_image_rgb=input_image_rgb,
                initial_rotation=initial_rotation_np,
                initial_gripper=initial_gripper,
                num_frames=num_frames
            )

        return output


# Pydantic models for request/response
class PredictRequest(BaseModel):
    prompt: str
    initial_rotation: List[float]  # [roll, pitch, yaw] in degrees
    initial_gripper: int  # 0 or 1
    num_frames: Optional[int] = None  # Uses server default if not specified


class PredictResponse(BaseModel):
    success: bool
    position: Optional[List[List[float]]] = None  # (num_frames-1 or num_frames, 3) in meters
    rotation: Optional[List[List[float]]] = None  # (num_frames-1, 3) in degrees
    gripper: Optional[List[int]] = None  # (num_frames-1,)
    error: Optional[str] = None


# Global server instance
server_instance: Optional[RoboWanServer] = None

# Create FastAPI app
app = FastAPI(title="RoboWan Server", description="FastAPI server for RoboWan multi-view inference")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global server_instance

    # Get parameters from environment variables
    lora_checkpoint = os.environ.get("LORA_CHECKPOINT")
    rot_grip_checkpoint = os.environ.get("ROT_GRIP_CHECKPOINT")
    model_base_path = os.environ.get("MODEL_BASE_PATH")
    wan_type = os.environ.get("WAN_TYPE", "5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP")
    use_dual_head = os.environ.get("USE_DUAL_HEAD", "true").lower() == "true"
    use_different_projection = os.environ.get("USE_DIFFERENT_PROJECTION", "false").lower() == "true"
    use_initial_gripper_state = os.environ.get("USE_INITIAL_GRIPPER_STATE", "false").lower() == "true"
    num_history_frames = int(os.environ.get("NUM_HISTORY_FRAMES", "1"))
    rotation_resolution = float(os.environ.get("ROTATION_RESOLUTION", "5.0"))
    hidden_dim = int(os.environ.get("HIDDEN_DIM", "512"))
    num_rotation_bins = int(os.environ.get("NUM_ROTATION_BINS", "72"))
    local_feat_size = int(os.environ.get("LOCAL_FEAT_SIZE", "5"))
    num_frames = int(os.environ.get("NUM_FRAMES", "13"))
    scene_bounds_str = os.environ.get("SCENE_BOUNDS", "0,-0.7,-0.05,0.8,0.7,0.65")
    scene_bounds = [float(x.strip()) for x in scene_bounds_str.split(',')]
    img_size = int(os.environ.get("IMG_SIZE", "256"))
    device = os.environ.get("DEVICE", "cuda:0")

    # Initialize server
    server_instance = RoboWanServer(
        lora_checkpoint=lora_checkpoint,
        rot_grip_checkpoint=rot_grip_checkpoint,
        model_base_path=model_base_path,
        wan_type=wan_type,
        use_dual_head=use_dual_head,
        rotation_resolution=rotation_resolution,
        hidden_dim=hidden_dim,
        num_rotation_bins=num_rotation_bins,
        local_feat_size=local_feat_size,
        num_frames=num_frames,
        scene_bounds=scene_bounds,
        img_size=img_size,
        device=device,
        use_different_projection=use_different_projection,
        use_initial_gripper_state=use_initial_gripper_state,
        num_history_frames=num_history_frames
    )

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Server initialized successfully!")


@app.post("/predict", response_model=PredictResponse)
async def predict_action(
    heatmap_images: List[UploadFile] = File(..., description="Multi-view heatmap images"),
    rgb_images: List[UploadFile] = File(..., description="Multi-view RGB images"),
    prompt: str = Form(..., description="Task instruction"),
    initial_rotation: str = Form(..., description="Initial rotation as comma-separated values"),
    initial_gripper: int = Form(..., description="Initial gripper state (0 or 1)"),
    num_frames: Optional[int] = Form(None, description="Number of frames to predict (uses server default if not specified)"),
    scene_bounds: Optional[str] = Form(None, description="Scene bounds as comma-separated values for consistency check")
):
    """
    Predict robot actions from observations
    """
    if server_instance is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 📥 Received prediction request")
        print(f"    - Prompt: {prompt}")
        print(f"    - Initial rotation: {initial_rotation}")
        print(f"    - Initial gripper: {initial_gripper}")
        print(f"    - Num frames: {num_frames if num_frames is not None else f'{server_instance.num_frames} (default)'}")
        print(f"    - Scene bounds (client): {scene_bounds}")

        # Parse initial rotation
        rotation_values = [float(x.strip()) for x in initial_rotation.split(',')]
        if len(rotation_values) != 3:
            raise ValueError("initial_rotation must have 3 values (roll, pitch, yaw)")

        # Check scene_bounds consistency
        if scene_bounds is not None:
            client_bounds = [float(x.strip()) for x in scene_bounds.split(',')]
            if len(client_bounds) != 6:
                raise ValueError("scene_bounds must have 6 values (x_min, y_min, z_min, x_max, y_max, z_max)")

            server_bounds = server_instance.inference_engine.scene_bounds

            # Compare with tolerance for floating point
            tolerance = 1e-6
            bounds_match = all(
                abs(c - s) < tolerance
                for c, s in zip(client_bounds, server_bounds)
            )

            if not bounds_match:
                error_msg = (
                    f"Scene bounds mismatch!\n"
                    f"  Client: {client_bounds}\n"
                    f"  Server: {server_bounds}\n"
                    f"Please ensure both client and server use the same scene_bounds."
                )
                print(f"    ❌ {error_msg}")
                raise ValueError(error_msg)
            else:
                print(f"    ✓ Scene bounds verified: {server_bounds}")

        # Read and decode images
        heatmap_imgs = []
        for img_file in heatmap_images:
            img_bytes = await img_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            heatmap_imgs.append(img)

        rgb_imgs = []
        for img_file in rgb_images:
            img_bytes = await img_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            rgb_imgs.append(img)

        print(f"    - Received {len(heatmap_imgs)} heatmap images")
        print(f"    - Received {len(rgb_imgs)} RGB images")

        # 判断是单帧模式还是多帧历史模式
        # 假设 num_views = 3
        # 单帧模式: 3张图 (num_views)
        # 多帧历史模式: num_views * num_history_frames 张图 (例如: 3 * 5 = 15张)
        num_views = 3  # 固定为3视角
        num_history_frames_server = server_instance.num_history_frames

        if len(heatmap_imgs) == num_views and len(rgb_imgs) == num_views:
            # 单帧模式
            print(f"    - Mode: Single-frame (num_views={num_views})")
            input_image = heatmap_imgs
            input_image_rgb = rgb_imgs
            input_images = None
            input_images_rgb = None

        elif len(heatmap_imgs) == num_views * num_history_frames_server and \
             len(rgb_imgs) == num_views * num_history_frames_server:
            # 多帧历史模式
            print(f"    - Mode: Multi-frame history (num_history={num_history_frames_server}, num_views={num_views})")

            # 重新组织图片: [img0, img1, ..., img14] -> [[img0, img1, img2], [img3, img4, img5], ...]
            input_images = []
            input_images_rgb = []

            for hist_idx in range(num_history_frames_server):
                start_idx = hist_idx * num_views
                end_idx = start_idx + num_views

                hist_heatmaps = heatmap_imgs[start_idx:end_idx]
                hist_rgbs = rgb_imgs[start_idx:end_idx]

                input_images.append(hist_heatmaps)
                input_images_rgb.append(hist_rgbs)

            # 兼容单帧接口：使用最后一个历史帧作为 input_image
            input_image = input_images[-1]  # List[PIL.Image] - num_views
            input_image_rgb = input_images_rgb[-1]  # List[PIL.Image] - num_views

        else:
            # 图片数量不匹配
            raise ValueError(
                f"Invalid number of images! Expected either:\n"
                f"  - Single-frame mode: {num_views} images (num_views)\n"
                f"  - Multi-frame history mode: {num_views * num_history_frames_server} images "
                f"(num_views={num_views} × num_history={num_history_frames_server})\n"
                f"  Got: {len(heatmap_imgs)} heatmap images, {len(rgb_imgs)} RGB images"
            )

        # Run prediction
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] 🤖 Running model inference...")
        output = server_instance.predict_action(
            prompt=prompt,
            input_image=input_image,
            input_image_rgb=input_image_rgb,
            input_images=input_images,
            input_images_rgb=input_images_rgb,
            initial_rotation=rotation_values,
            initial_gripper=initial_gripper,
            num_frames=num_frames
        )

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] ✓ Prediction completed")

        # Convert to lists
        position_list = output['position'].tolist()
        rotation_list = output['rotation'].tolist()
        gripper_list = output['gripper'].tolist()

        return PredictResponse(
            success=True,
            position=position_list,
            rotation=rotation_list,
            gripper=gripper_list
        )

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] ❌ Error: {e}")
        traceback.print_exc()
        return PredictResponse(
            success=False,
            error=str(e)
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if server_instance is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/config")
async def get_config():
    """
    Get server configuration for client verification

    Returns:
        Dictionary containing server configuration:
            - use_different_projection: bool
            - scene_bounds: List[float]
            - img_size: int
            - num_frames: int
    """
    if server_instance is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return {
        "use_different_projection": server_instance.inference_engine.use_different_projection,
        "scene_bounds": server_instance.inference_engine.scene_bounds,
        "img_size": server_instance.inference_engine.img_size,
        "num_frames": server_instance.num_frames,
        "timestamp": datetime.now().isoformat()
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RoboWan FastAPI Server")
    parser.add_argument("--lora_checkpoint", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--rot_grip_checkpoint", type=str, required=True, help="Path to rotation/gripper checkpoint")
    parser.add_argument("--model_base_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--wan_type", type=str, default="5B_TI2V_RGB_HEATMAP_MV_ROT_GRIP", help="Model type")
    parser.add_argument("--use_dual_head", action="store_true", default=True, help="Use dual head mode")
    parser.add_argument("--use_different_projection", action="store_true", default=False, help="Use different projection mode (3cam_different_projection)")
    parser.add_argument("--use_initial_gripper_state", action="store_true", default=False, help="Use initial gripper state as input (must match training)")
    parser.add_argument("--num_history_frames", type=int, default=1, help="Number of history frames (must match training, allowed: 1, 2, or 1+4N like 5, 9, 13...)")
    parser.add_argument("--rotation_resolution", type=float, default=5.0, help="Rotation resolution in degrees")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_rotation_bins", type=int, default=72, help="Number of rotation bins")
    parser.add_argument("--local_feat_size", type=int, default=5, help="Local feature size for rotation/gripper predictor")
    parser.add_argument("--num_frames", type=int, default=13, help="Total number of frames (including initial frame)")
    parser.add_argument("--scene_bounds", type=str, default="0,-0.7,-0.05,0.8,0.7,0.65",
                        help="Scene bounds as comma-separated values: x_min,y_min,z_min,x_max,y_max,z_max")
    parser.add_argument("--img_size", type=int, default=256, help="Image size for projection interface")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on")
    args = parser.parse_args()

    # Set environment variables for startup event
    os.environ["LORA_CHECKPOINT"] = args.lora_checkpoint
    os.environ["ROT_GRIP_CHECKPOINT"] = args.rot_grip_checkpoint
    os.environ["MODEL_BASE_PATH"] = args.model_base_path
    os.environ["WAN_TYPE"] = args.wan_type
    os.environ["USE_DUAL_HEAD"] = str(args.use_dual_head).lower()
    os.environ["USE_DIFFERENT_PROJECTION"] = str(args.use_different_projection).lower()
    os.environ["USE_INITIAL_GRIPPER_STATE"] = str(args.use_initial_gripper_state).lower()
    os.environ["NUM_HISTORY_FRAMES"] = str(args.num_history_frames)
    os.environ["ROTATION_RESOLUTION"] = str(args.rotation_resolution)
    os.environ["HIDDEN_DIM"] = str(args.hidden_dim)
    os.environ["NUM_ROTATION_BINS"] = str(args.num_rotation_bins)
    os.environ["LOCAL_FEAT_SIZE"] = str(args.local_feat_size)
    os.environ["NUM_FRAMES"] = str(args.num_frames)
    os.environ["SCENE_BOUNDS"] = args.scene_bounds
    os.environ["IMG_SIZE"] = str(args.img_size)
    os.environ["DEVICE"] = args.device

    # Run server
    print(f"Starting RoboWan FastAPI server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
