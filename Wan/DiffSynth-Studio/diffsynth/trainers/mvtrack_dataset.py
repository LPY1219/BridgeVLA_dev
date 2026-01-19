"""
MVTrack Dataset for Multi-View Video Diffusion Model Pretraining

This dataset is designed for pretraining multi-view video diffusion models using the MVTrack dataset.
The MVTrack dataset contains multi-view tracking videos where:
- Each scene folder contains multiple view subfolders (e.g., bag1-1, bag1-2, bag1-3, bag1-4)
- Each view folder contains:
  - img/: RGB images sequence (00001.jpg, 00002.jpg, ...)
  - groundtruth.txt: Bounding box annotations (x, y, w, h) for each frame
  - invisible.txt: Frames where the object is not visible (NOT USED - we use all frames with bbox)
- The folder naming format is "object_name+number" (e.g., bag1, basketball3)
- Task instruction is generated as "Track the {object}"

Key differences from Franka dataset:
1. No point cloud data - heatmaps are generated from bbox centers
2. No robot poses/gripper states - this is a tracking dataset
3. Different directory structure
4. Multiple views per scene, randomly sample 3 views for training
"""

import os
import re
import random
import glob
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Any
from copy import deepcopy
import matplotlib.pyplot as plt

from .unified_dataset import UnifiedDataset


def parse_object_name_from_folder(folder_name: str) -> str:
    """
    Parse object name from folder name.
    The folder naming format is "object_name+number" (e.g., bag1, basketball3)

    Args:
        folder_name: Folder name like 'bag1', 'basketball3', 'keyboard3'

    Returns:
        Object name like 'bag', 'basketball', 'keyboard'
    """
    # Match letters followed by numbers
    match = re.match(r'^([a-zA-Z]+)\d+$', folder_name)
    if match:
        return match.group(1)
    # If no number suffix, return the whole name
    return folder_name


def load_groundtruth(gt_path: str) -> np.ndarray:
    """
    Load groundtruth.txt file containing bounding box annotations.
    Each line contains: x, y, w, h (top-left corner and width/height)

    Args:
        gt_path: Path to groundtruth.txt file

    Returns:
        numpy array of shape (N, 4) where N is number of frames
        Each row is [x, y, w, h]
    """
    bboxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Parse comma-separated values
                parts = line.split(',')
                if len(parts) >= 4:
                    x, y, w, h = map(float, parts[:4])
                    bboxes.append([x, y, w, h])
    return np.array(bboxes)


def is_valid_bbox(bbox: np.ndarray, min_size: float = 1.0) -> bool:
    """
    Check if a bounding box is valid (not 0,0,0,0 and has reasonable size).

    Args:
        bbox: Bounding box [x, y, w, h]
        min_size: Minimum width/height to be considered valid

    Returns:
        True if bbox is valid, False otherwise
    """
    x, y, w, h = bbox
    # Check if bbox is (0,0,0,0) or has zero/negative dimensions
    if w <= min_size or h <= min_size:
        return False
    # Check if all values are zero (invalid annotation)
    if x == 0 and y == 0 and w == 0 and h == 0:
        return False
    return True


def get_valid_frame_indices(bboxes: np.ndarray, min_size: float = 1.0) -> List[int]:
    """
    Get indices of frames with valid bounding boxes.

    Args:
        bboxes: Array of bounding boxes (N, 4)
        min_size: Minimum width/height to be considered valid

    Returns:
        List of valid frame indices
    """
    valid_indices = []
    for i, bbox in enumerate(bboxes):
        if is_valid_bbox(bbox, min_size):
            valid_indices.append(i)
    return valid_indices


def generate_heatmap_from_bbox_center(
    bbox: np.ndarray,
    height: int,
    width: int,
    sigma: float = 5.0
) -> np.ndarray:
    """
    Generate a Gaussian heatmap centered at the bounding box center.

    Args:
        bbox: Bounding box [x, y, w, h] (top-left corner and width/height)
        height: Image height
        width: Image width
        sigma: Standard deviation of Gaussian

    Returns:
        Heatmap array of shape (height, width)
    """
    x, y, w, h = bbox

    # Calculate center point
    center_x = x + w / 2
    center_y = y + h / 2

    # Create coordinate grids
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    # Calculate Gaussian distribution
    heatmap = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))

    # Normalize to [0, 1]
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap.astype(np.float32)


def heatmap_to_colormap_pil(heatmap: np.ndarray) -> Image.Image:
    """
    Convert a single heatmap to a colored PIL image using JET colormap.
    This matches the format used in the Franka dataset preprocessing.

    Args:
        heatmap: Heatmap array of shape (H, W), values in [0, 1]

    Returns:
        PIL Image with JET colormap applied
    """
    # Normalize to [0, 1]
    hm_min = heatmap.min()
    hm_max = heatmap.max()
    if hm_max > hm_min:
        hm_norm = (heatmap - hm_min) / (hm_max - hm_min)
    else:
        hm_norm = heatmap

    # Apply JET colormap (consistent with Franka dataset)
    hm_uint8 = (hm_norm * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)

    return Image.fromarray(hm_colored)


def draw_bbox_on_image(image: Image.Image, bbox: np.ndarray, color: str = 'green', width: int = 2) -> Image.Image:
    """
    Draw bounding box on PIL image.

    Args:
        image: PIL Image
        bbox: Bounding box [x, y, w, h]
        color: Box color
        width: Line width

    Returns:
        PIL Image with bbox drawn
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    x, y, w, h = bbox
    # Draw rectangle
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    return img_copy


class MVTrackDataset(Dataset):
    """
    Multi-View Tracking Dataset for video diffusion model pretraining.

    This dataset loads multi-view tracking videos and generates training samples
    with RGB images and corresponding heatmaps based on bounding box annotations.

    NOTE: We use ALL frames that have valid bounding box annotations,
    regardless of the invisible.txt file.
    """

    def __init__(
        self,
        data_root: str,
        split_file: str,  # Path to train_split.txt or val_split.txt
        sequence_length: int = 12,
        step_interval: int = 1,
        min_sequence_length: int = 15,
        image_size: Tuple[int, int] = (256, 256),
        sigma: float = 5.0,
        num_views: int = 3,
        augmentation: bool = True,
        debug: bool = False,
        colormap_name: str = 'jet',
    ):
        """
        Initialize MVTrack dataset.

        Args:
            data_root: Root directory of MVTrack dataset
            split_file: Path to split file (train_split.txt or val_split.txt)
            sequence_length: Number of future frames to predict (not including first frame)
            step_interval: Frame sampling interval
            min_sequence_length: Minimum sequence length requirement
            image_size: Output image size (H, W)
            sigma: Standard deviation for Gaussian heatmap generation
            num_views: Number of views to sample for each training example
            augmentation: Whether to use data augmentation
            debug: Debug mode (use fewer samples)
            colormap_name: Colormap name (default 'jet' for consistency)
        """
        self.data_root = data_root
        self.split_file = split_file
        self.sequence_length = sequence_length
        self.step_interval = step_interval
        self.min_sequence_length = min_sequence_length
        self.image_size = image_size
        self.sigma = sigma
        self.num_views = num_views
        self.augmentation = augmentation
        self.debug = debug
        self.colormap_name = colormap_name

        # Load split file and organize samples
        self.samples = self._load_and_organize_samples()

        print(f"MVTrackDataset initialized with {len(self.samples)} valid samples")
        print(f"  - Sequence length: {sequence_length}")
        print(f"  - Image size: {image_size}")
        print(f"  - Number of views: {num_views}")
        print(f"  - Sigma: {sigma}")

    def _load_and_organize_samples(self) -> List[Dict]:
        """
        Load split file and organize samples by scene.

        NOTE: We use ALL frames that have valid bounding box annotations,
        regardless of the invisible.txt file.

        Returns:
            List of sample dictionaries containing:
            - scene_name: Scene folder name (e.g., 'bag1')
            - object_name: Parsed object name (e.g., 'bag')
            - view_folders: List of available view folders for this scene
            - valid_start_indices: List of valid starting frame indices
        """
        samples = []

        # Read split file
        with open(self.split_file, 'r') as f:
            view_list = [line.strip() for line in f if line.strip()]

        # Group views by scene
        scene_views = {}
        for view_name in view_list:
            # Parse scene name from view name (e.g., 'bag1-1' -> 'bag1')
            parts = view_name.rsplit('-', 1)
            if len(parts) == 2:
                scene_name = parts[0]
            else:
                scene_name = view_name

            if scene_name not in scene_views:
                scene_views[scene_name] = []
            scene_views[scene_name].append(view_name)

        # Create samples for each scene
        for scene_name, views in scene_views.items():
            # Skip if not enough views
            if len(views) < self.num_views:
                print(f"  Skipping {scene_name}: only {len(views)} views (need {self.num_views})")
                continue

            # Check each view folder and find valid samples
            view_info_list = []
            min_num_frames = float('inf')

            for view_name in views:
                scene_folder = scene_name  # scene_name is the parent folder
                view_folder = os.path.join(self.data_root, scene_folder, view_name)

                if not os.path.exists(view_folder):
                    continue

                img_folder = os.path.join(view_folder, 'img')
                gt_file = os.path.join(view_folder, 'groundtruth.txt')

                if not os.path.exists(img_folder) or not os.path.exists(gt_file):
                    continue

                # Count images and load annotations
                images = sorted(glob.glob(os.path.join(img_folder, '*.jpg')))
                if len(images) == 0:
                    continue

                bboxes = load_groundtruth(gt_file)

                # Get valid frame indices (frames with valid bounding boxes)
                num_total_frames = min(len(images), len(bboxes))
                valid_frame_indices = get_valid_frame_indices(bboxes[:num_total_frames])

                # Need enough valid frames
                if len(valid_frame_indices) < self.min_sequence_length:
                    continue

                view_info_list.append({
                    'view_name': view_name,
                    'view_folder': view_folder,
                    'images': images,
                    'bboxes': bboxes,
                    'num_frames': num_total_frames,
                    'valid_frame_indices': valid_frame_indices,  # Only frames with valid bbox
                })

                min_num_frames = min(min_num_frames, len(valid_frame_indices))

            if len(view_info_list) < self.num_views:
                continue

            # Calculate valid start indices
            # Need sequence_length * step_interval + 1 frames from start
            required_frames = self.sequence_length * self.step_interval + 1

            if min_num_frames < required_frames:
                continue

            # Create samples
            # Find the maximum common valid start index
            max_start_idx = min_num_frames - required_frames

            if max_start_idx < 0:
                continue

            object_name = parse_object_name_from_folder(scene_name)

            # In debug mode, only add one sample per scene
            if self.debug:
                samples.append({
                    'scene_name': scene_name,
                    'object_name': object_name,
                    'view_info_list': view_info_list,
                    'max_start_idx': max_start_idx,
                })
                if len(samples) >= 5:  # Limit to 5 scenes in debug mode
                    break
            else:
                # Add multiple samples per scene based on different start indices
                # Sample every step_interval frames for diversity
                for start_idx in range(0, max_start_idx + 1, self.step_interval):
                    samples.append({
                        'scene_name': scene_name,
                        'object_name': object_name,
                        'view_info_list': view_info_list,
                        'start_idx': start_idx,
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_sample_data(self, idx: int, max_retries: int = 10) -> Dict[str, Any]:
        """
        Internal method to get sample data with retry logic for invalid bboxes.

        Args:
            idx: Sample index
            max_retries: Maximum number of retries if invalid bbox is encountered

        Returns:
            Sample data dictionary
        """
        for retry in range(max_retries):
            sample = self.samples[idx]
            scene_name = sample['scene_name']
            object_name = sample['object_name']
            view_info_list = sample['view_info_list']

            # Get start index within valid frames
            if 'start_idx' in sample:
                start_idx = sample['start_idx']
            else:
                # Random start index for debug mode
                start_idx = random.randint(0, sample['max_start_idx'])

            # Randomly select num_views views
            selected_views = random.sample(view_info_list, self.num_views)

            # Generate frame indices within valid frames
            # These are indices into the valid_frame_indices list, not actual frame indices
            valid_frame_positions = [start_idx + i * self.step_interval for i in range(self.sequence_length + 1)]

            # Load data for each view
            rgb_frames_all_views = []
            heatmap_frames_all_views = []
            has_invalid_bbox = False

            # Initialize with empty lists for each time step
            for _ in range(self.sequence_length + 1):
                rgb_frames_all_views.append([])
                heatmap_frames_all_views.append([])

            for view_info in selected_views:
                images = view_info['images']
                bboxes = view_info['bboxes']
                valid_frame_indices = view_info['valid_frame_indices']

                for t_idx, valid_pos in enumerate(valid_frame_positions):
                    # Map position to actual frame index using valid_frame_indices
                    if valid_pos >= len(valid_frame_indices):
                        valid_pos = len(valid_frame_indices) - 1
                    actual_frame_idx = valid_frame_indices[valid_pos]

                    # Double check bbox is valid (should be, but verify)
                    bbox = bboxes[actual_frame_idx]
                    if not is_valid_bbox(bbox):
                        has_invalid_bbox = True
                        break

                    # Load RGB image
                    img_path = images[actual_frame_idx]
                    rgb_img = Image.open(img_path).convert('RGB')
                    orig_w, orig_h = rgb_img.size
                    rgb_img = rgb_img.resize(self.image_size, Image.BILINEAR)

                    # Scale bbox to image size
                    scale_x = self.image_size[0] / orig_w
                    scale_y = self.image_size[1] / orig_h
                    scaled_bbox = np.array([
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ])

                    # Generate heatmap
                    heatmap = generate_heatmap_from_bbox_center(
                        scaled_bbox,
                        self.image_size[1],  # height
                        self.image_size[0],  # width
                        sigma=self.sigma
                    )
                    heatmap_pil = heatmap_to_colormap_pil(heatmap)

                    rgb_frames_all_views[t_idx].append(rgb_img)
                    heatmap_frames_all_views[t_idx].append(heatmap_pil)

                if has_invalid_bbox:
                    break

            if has_invalid_bbox:
                # Try a different sample
                idx = random.randint(0, len(self.samples) - 1)
                continue

            # Generate instruction
            instruction = f"Track the {object_name}"

            # Construct output dictionary
            data = {
                'prompt': instruction,
                'video': heatmap_frames_all_views,
                'input_image': deepcopy(heatmap_frames_all_views[0]),
                'input_image_rgb': deepcopy(rgb_frames_all_views[0]),
                'input_video_rgb': rgb_frames_all_views,
            }

            return data

        # If all retries failed, return the last attempt anyway
        print(f"Warning: Could not find valid sample after {max_retries} retries")
        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
            - prompt: Task instruction ("Track the {object}")
            - video: List[List[PIL.Image]] - (T+1) x num_views heatmap images
            - input_image: List[PIL.Image] - First frame heatmaps for each view
            - input_image_rgb: List[PIL.Image] - First frame RGB images for each view
            - input_video_rgb: List[List[PIL.Image]] - (T+1) x num_views RGB images
        """
        return self._get_sample_data(idx)

    def get_sample_with_bbox(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample with bounding box information for visualization.

        Returns:
            Dictionary containing all data from __getitem__ plus:
            - bboxes: List[List[np.ndarray]] - (T+1) x num_views scaled bounding boxes
            - rgb_with_bbox: List[List[PIL.Image]] - (T+1) x num_views RGB images with bbox drawn
        """
        sample = self.samples[idx]
        scene_name = sample['scene_name']
        object_name = sample['object_name']
        view_info_list = sample['view_info_list']

        # Get start index
        if 'start_idx' in sample:
            start_idx = sample['start_idx']
        else:
            start_idx = random.randint(0, sample['max_start_idx'])

        # Randomly select num_views views
        selected_views = random.sample(view_info_list, self.num_views)

        # Generate frame positions within valid frames
        valid_frame_positions = [start_idx + i * self.step_interval for i in range(self.sequence_length + 1)]

        # Load data for each view
        rgb_frames_all_views = []
        heatmap_frames_all_views = []
        bbox_all_views = []
        rgb_with_bbox_all_views = []

        # Initialize with empty lists for each time step
        for _ in range(self.sequence_length + 1):
            rgb_frames_all_views.append([])
            heatmap_frames_all_views.append([])
            bbox_all_views.append([])
            rgb_with_bbox_all_views.append([])

        for view_info in selected_views:
            images = view_info['images']
            bboxes = view_info['bboxes']
            valid_frame_indices = view_info['valid_frame_indices']

            for t_idx, valid_pos in enumerate(valid_frame_positions):
                # Map position to actual frame index using valid_frame_indices
                if valid_pos >= len(valid_frame_indices):
                    valid_pos = len(valid_frame_indices) - 1
                actual_frame_idx = valid_frame_indices[valid_pos]

                # Load RGB image
                img_path = images[actual_frame_idx]
                rgb_img = Image.open(img_path).convert('RGB')
                orig_w, orig_h = rgb_img.size
                rgb_img = rgb_img.resize(self.image_size, Image.BILINEAR)

                # Get bbox and scale
                bbox = bboxes[actual_frame_idx]
                scale_x = self.image_size[0] / orig_w
                scale_y = self.image_size[1] / orig_h
                scaled_bbox = np.array([
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                ])

                # Generate heatmap
                heatmap = generate_heatmap_from_bbox_center(
                    scaled_bbox,
                    self.image_size[1],
                    self.image_size[0],
                    sigma=self.sigma
                )
                heatmap_pil = heatmap_to_colormap_pil(heatmap)

                # Draw bbox on RGB image
                rgb_with_bbox = draw_bbox_on_image(rgb_img, scaled_bbox, color='lime', width=2)

                rgb_frames_all_views[t_idx].append(rgb_img)
                heatmap_frames_all_views[t_idx].append(heatmap_pil)
                bbox_all_views[t_idx].append(scaled_bbox)
                rgb_with_bbox_all_views[t_idx].append(rgb_with_bbox)

        instruction = f"Track the {object_name}"

        data = {
            'prompt': instruction,
            'video': heatmap_frames_all_views,
            'input_image': deepcopy(heatmap_frames_all_views[0]),
            'input_image_rgb': deepcopy(rgb_frames_all_views[0]),
            'input_video_rgb': rgb_frames_all_views,
            'bboxes': bbox_all_views,
            'rgb_with_bbox': rgb_with_bbox_all_views,
            'scene_name': scene_name,
        }

        return data


class MVTrackUnifiedDataset(UnifiedDataset):
    """
    Wrapper class to make MVTrackDataset compatible with UnifiedDataset interface.
    """

    def __init__(
        self,
        mvtrack_config: Dict[str, Any],
        colormap_name: str = 'jet',
        repeat: int = 1,
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV",
        **kwargs
    ):
        """
        Initialize MVTrack unified dataset wrapper.

        Args:
            mvtrack_config: Configuration for MVTrackDataset
            colormap_name: Colormap name for heatmap visualization
            repeat: Dataset repetition factor
            wan_type: Wan model type
            **kwargs: Additional arguments for UnifiedDataset
        """
        self.colormap_name = colormap_name
        self.mvtrack_config = mvtrack_config
        self.wan_type = wan_type

        # Create MVTrack dataset
        self.mvtrack_dataset = MVTrackDataset(**mvtrack_config)

        # Initialize parent class with dummy parameters
        super().__init__(
            base_path="",
            metadata_path=None,
            repeat=repeat,
            data_file_keys=(),
            main_data_operator=lambda x: x,
            **kwargs
        )

        # Reset data
        self.data = []
        self.cached_data = []
        self.load_from_cache = False

        print(f"MVTrackUnifiedDataset initialized with {len(self.mvtrack_dataset)} samples")

    def load_metadata(self, metadata_path):
        """Override to skip metadata loading."""
        pass

    def __len__(self) -> int:
        return len(self.mvtrack_dataset) * self.repeat

    def __getitem__(self, data_id: int) -> Dict[str, Any]:
        """
        Get a sample in Wan training format.
        """
        actual_idx = data_id % len(self.mvtrack_dataset)
        return self.mvtrack_dataset[actual_idx]


class MVTrackDatasetFactory:
    """
    Factory class for creating MVTrack datasets with different configurations.
    """

    @staticmethod
    def create_mvtrack_dataset(
        data_root: str,
        split_files: List[str] = None,
        sequence_length: int = 12,
        step_interval: int = 1,
        min_sequence_length: int = 15,
        image_size: Tuple[int, int] = (256, 256),
        sigma: float = 5.0,
        num_views: int = 3,
        augmentation: bool = True,
        debug: bool = False,
        colormap_name: str = 'jet',
        repeat: int = 1,
        wan_type: str = "5B_TI2V_RGB_HEATMAP_MV",
        **kwargs
    ) -> MVTrackUnifiedDataset:
        """
        Create MVTrack dataset for pretraining.

        Args:
            data_root: Root directory of MVTrack dataset
            split_files: List of split files to use (default: train_split.txt + val_split.txt)
            sequence_length: Number of future frames to predict
            step_interval: Frame sampling interval
            min_sequence_length: Minimum sequence length requirement
            image_size: Output image size (H, W)
            sigma: Standard deviation for Gaussian heatmap
            num_views: Number of views to sample
            augmentation: Whether to use data augmentation
            debug: Debug mode
            colormap_name: Colormap name
            repeat: Dataset repetition factor
            wan_type: Wan model type
            **kwargs: Additional arguments

        Returns:
            MVTrackUnifiedDataset instance
        """
        # Default split files
        if split_files is None:
            split_files = [
                os.path.join(data_root, 'train_split.txt'),
                os.path.join(data_root, 'val_split.txt'),
            ]

        # Create config
        mvtrack_config = {
            'data_root': data_root,
            'split_file': split_files[0],  # Will be handled in multi-split case
            'sequence_length': sequence_length,
            'step_interval': step_interval,
            'min_sequence_length': min_sequence_length,
            'image_size': image_size,
            'sigma': sigma,
            'num_views': num_views,
            'augmentation': augmentation,
            'debug': debug,
            'colormap_name': colormap_name,
        }

        # Handle multiple split files by concatenating
        if len(split_files) > 1:
            # Merge all split files into a temporary combined file
            combined_views = []
            for split_file in split_files:
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        combined_views.extend([line.strip() for line in f if line.strip()])

            # Write to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write('\n'.join(combined_views))
                combined_split_file = f.name

            mvtrack_config['split_file'] = combined_split_file
            print(f"Combined {len(split_files)} split files into {combined_split_file}")
            print(f"Total views: {len(combined_views)}")

        return MVTrackUnifiedDataset(
            mvtrack_config=mvtrack_config,
            colormap_name=colormap_name,
            repeat=repeat,
            wan_type=wan_type,
            **kwargs
        )


def visualize_sample(sample: Dict[str, Any], save_path: str = None, show: bool = True):
    """
    Visualize a sample with RGB video (with bbox) and heatmap video.

    Args:
        sample: Sample dictionary from get_sample_with_bbox()
        save_path: Path to save the visualization image
        show: Whether to display the image
    """
    video_heatmap = sample['video']  # (T+1) x num_views
    rgb_with_bbox = sample['rgb_with_bbox']  # (T+1) x num_views
    prompt = sample['prompt']
    scene_name = sample.get('scene_name', 'unknown')

    num_frames = len(video_heatmap)
    num_views = len(video_heatmap[0])

    # Create figure: 2 rows (RGB+bbox, Heatmap) x (num_frames * num_views) columns
    # Actually, let's organize as: rows = frames, cols = views * 2 (RGB, Heatmap)
    fig, axes = plt.subplots(num_frames, num_views * 2, figsize=(num_views * 6, num_frames * 3))

    if num_frames == 1:
        axes = axes.reshape(1, -1)

    for t in range(num_frames):
        for v in range(num_views):
            # RGB with bbox
            ax_rgb = axes[t, v * 2]
            ax_rgb.imshow(np.array(rgb_with_bbox[t][v]))
            ax_rgb.axis('off')
            if t == 0:
                ax_rgb.set_title(f'RGB View {v}', fontsize=10)
            if v == 0:
                ax_rgb.set_ylabel(f'T={t}', fontsize=10)

            # Heatmap
            ax_hm = axes[t, v * 2 + 1]
            ax_hm.imshow(np.array(video_heatmap[t][v]))
            ax_hm.axis('off')
            if t == 0:
                ax_hm.set_title(f'Heatmap View {v}', fontsize=10)

    plt.suptitle(f'Scene: {scene_name} | Prompt: "{prompt}"', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# Test code
if __name__ == "__main__":
    print("Testing MVTrack Dataset...")
    print("="*60)

    data_root = "/data/lpy/pretrain_data/Dataset/MVTrack"
    save_dir = "/home/lpy/BridgeVLA_dev/Wan/DiffSynth-Studio/examples/debug_log/pretrain_dataset"

    # Number of samples to visualize
    NUM_SAMPLES_TO_VISUALIZE = 5

    if os.path.exists(data_root):
        # Test dataset creation
        print("\n1. Creating dataset...")
        dataset = MVTrackDatasetFactory.create_mvtrack_dataset(
            data_root=data_root,
            sequence_length=12,
            step_interval=2,
            min_sequence_length=20,
            image_size=(256, 256),
            sigma=5.0,
            num_views=3,
            debug=True,
            repeat=1,
        )

        print(f"\nDataset created with {len(dataset)} samples")

        # Test getting a sample
        if len(dataset) > 0:
            print("\n2. Getting sample info...")
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Prompt: {sample['prompt']}")
            print(f"Video frames: {len(sample['video'])}")
            print(f"Views per frame: {len(sample['video'][0])}")
            print(f"Input image views: {len(sample['input_image'])}")
            print(f"Input image RGB views: {len(sample['input_image_rgb'])}")

            # Check image sizes
            if len(sample['video']) > 0 and len(sample['video'][0]) > 0:
                print(f"First heatmap size: {sample['video'][0][0].size}")
            if len(sample['input_video_rgb']) > 0 and len(sample['input_video_rgb'][0]) > 0:
                print(f"First RGB size: {sample['input_video_rgb'][0][0].size}")

            # Visualization with bbox for multiple samples
            print(f"\n3. Creating visualization for {NUM_SAMPLES_TO_VISUALIZE} samples...")
            os.makedirs(save_dir, exist_ok=True)

            num_samples = min(NUM_SAMPLES_TO_VISUALIZE, len(dataset))

            for sample_idx in range(num_samples):
                print(f"\n  Processing sample {sample_idx + 1}/{num_samples}...")

                # Get sample with bbox info
                sample_with_bbox = dataset.mvtrack_dataset.get_sample_with_bbox(sample_idx)
                scene_name = sample_with_bbox.get('scene_name', f'sample_{sample_idx}')
                prompt = sample_with_bbox['prompt']

                print(f"    Scene: {scene_name}")
                print(f"    Prompt: {prompt}")

                # Create sample directory
                sample_dir = os.path.join(save_dir, f"sample_{sample_idx:03d}_{scene_name}")
                os.makedirs(sample_dir, exist_ok=True)

                # Visualize full video
                visualize_sample(
                    sample_with_bbox,
                    save_path=os.path.join(sample_dir, "full_visualization.png"),
                    show=False
                )

                # Save individual frames
                num_frames_to_save = min(5, len(sample_with_bbox['video']))  # Save first 5 frames
                for t in range(num_frames_to_save):
                    for v in range(len(sample_with_bbox['video'][t])):
                        # Save heatmap
                        hm_path = os.path.join(sample_dir, f"frame{t:02d}_view{v}_heatmap.png")
                        sample_with_bbox['video'][t][v].save(hm_path)

                        # Save RGB with bbox
                        rgb_path = os.path.join(sample_dir, f"frame{t:02d}_view{v}_rgb_bbox.png")
                        sample_with_bbox['rgb_with_bbox'][t][v].save(rgb_path)

                        # Save RGB without bbox
                        rgb_clean_path = os.path.join(sample_dir, f"frame{t:02d}_view{v}_rgb.png")
                        sample_with_bbox['input_video_rgb'][t][v].save(rgb_clean_path)

                print(f"    Saved to: {sample_dir}")

            print(f"\n4. Summary:")
            print(f"   - Total samples visualized: {num_samples}")
            print(f"   - Visualization directory: {save_dir}")
            print(f"   - Each sample directory contains:")
            print(f"     * full_visualization.png: Full video grid")
            print(f"     * frame*_view*_heatmap.png: Individual heatmap frames")
            print(f"     * frame*_view*_rgb_bbox.png: RGB frames with bounding box")
            print(f"     * frame*_view*_rgb.png: Clean RGB frames")

    else:
        print(f"Data root not found: {data_root}")

    print("\n" + "="*60)
    print("Test completed!")
