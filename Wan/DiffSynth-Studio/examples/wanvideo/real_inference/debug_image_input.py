"""
Debug script for visualizing RGB and Heatmap input images
Extracted from RoboWan_client_3zed.py for standalone debugging

Usage:
    python debug_image_input.py                    # Capture and visualize once
    python debug_image_input.py --continuous       # Continuous capture mode
    python debug_image_input.py --save-dir ./debug # Save images to directory
"""

# Disable proxy
import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

# OpenMP settings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Preload libgomp
import ctypes
try:
    ctypes.CDLL('libgomp.so.1', mode=ctypes.RTLD_GLOBAL)
except:
    pass

# Import ZED before PyTorch
try:
    import pyzed.sl as sl
except:
    pass

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from datetime import datetime
import sys
import argparse
import open3d as o3d

# Add paths
diffsynth_path = "/media/casia/data4/lpy/RoboWan/BridgeVLA_dev/Wan/DiffSynth-Studio"
sys.path.insert(0, diffsynth_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_collection_path = os.path.join(current_dir, "../../../../../data_collection")
data_collection_path = os.path.abspath(data_collection_path)
sys.path.insert(0, data_collection_path)

# ====================== Configuration ======================
USE_DIFFERENT_PROJECTION = True
SCENE_BOUNDS = [-0.1, -0.5, -0.1, 0.9, 0.5, 0.9]
IMG_SIZE = 256
USE_MERGED_POINTCLOUD = False


def get_projection_interface_class(use_different_projection: bool):
    """Get ProjectionInterface class based on projection mode"""
    if use_different_projection:
        from diffsynth.trainers.base_multi_view_dataset_with_rot_grip_3cam_different_projection import (
            ProjectionInterface,
            build_extrinsic_matrix,
            convert_pcd_to_base,
            _norm_rgb
        )
        print("[ProjectionInterface] Using DIFFERENT projection mode")
    else:
        from diffsynth.trainers.base_multi_view_dataset_with_rot_grip_3cam import (
            ProjectionInterface,
            build_extrinsic_matrix,
            convert_pcd_to_base,
            _norm_rgb
        )
        print("[ProjectionInterface] Using DEFAULT projection mode")
    return ProjectionInterface, build_extrinsic_matrix, convert_pcd_to_base, _norm_rgb


class ImageDebugger:
    """Debug class for visualizing RGB and Heatmap images"""

    def __init__(self, img_size: int = 256, scene_bounds: list = None, sigma: float = 1.5):
        """
        Initialize debugger

        Args:
            img_size: Image size
            scene_bounds: Scene bounds [x_min, y_min, z_min, x_max, y_max, z_max]
            sigma: Heatmap gaussian sigma
        """
        self.img_size = (img_size, img_size)
        self.scene_bounds = scene_bounds if scene_bounds is not None else SCENE_BOUNDS
        self.sigma = sigma
        self.use_different_projection = USE_DIFFERENT_PROJECTION
        self.use_merged_pointcloud = USE_MERGED_POINTCLOUD

        # Import projection interface
        ProjectionInterface, build_extrinsic_matrix, convert_pcd_to_base, self._norm_rgb = \
            get_projection_interface_class(USE_DIFFERENT_PROJECTION)
        self.convert_pcd_to_base = convert_pcd_to_base

        # Import camera extrinsics
        from real_camera_utils_lpy import get_cam_extrinsic
        self.get_cam_extrinsic = get_cam_extrinsic

        # Initialize extrinsic matrices
        self.extrinsic_matrix_1 = get_cam_extrinsic("3rd_1")
        self.extrinsic_matrix_2 = get_cam_extrinsic("3rd_2")
        self.extrinsic_matrix_3 = get_cam_extrinsic("3rd_3")

        # Initialize projection interface
        self.projection_interface = ProjectionInterface(
            img_size=img_size,
            rend_three_views=True,
            add_depth=False
        )

        print(f"[ImageDebugger] Initialized with:")
        print(f"  - Image size: {img_size}")
        print(f"  - Scene bounds: {self.scene_bounds}")
        print(f"  - Sigma: {sigma}")
        print(f"  - Projection mode: {'DIFFERENT' if USE_DIFFERENT_PROJECTION else 'DEFAULT'}")

    def preprocess(self, pcd_list, feat_list, all_poses: np.ndarray):
        """
        Preprocess point clouds and poses (same as RoboWanClient.preprocess)

        Args:
            pcd_list: List of point clouds [[pcd_cam1, pcd_cam2, pcd_cam3]]
            feat_list: List of features [[feat_cam1, feat_cam2, feat_cam3]]
            all_poses: Pose array [num_poses, 7] - (x,y,z,w,x,y,z) wxyz format

        Returns:
            processed_pcd, processed_rgb, wpt_local, rot_xyzw, rev_trans
        """
        import bridgevla.mvt.utils as mvt_utils

        if not isinstance(pcd_list, list):
            pcd_list = [pcd_list]
        if not isinstance(feat_list, list):
            feat_list = [feat_list]

        num_frames = len(pcd_list)
        merged_pcd_list = []
        merged_feat_list = []

        for frame_idx in range(num_frames):
            frame_pcds = pcd_list[frame_idx]
            frame_feats = feat_list[frame_idx]

            # Normalize RGB
            frame_feats_norm = [self._norm_rgb(feat) for feat in frame_feats]

            # Apply extrinsic transforms
            pcd_cam1_base = self.convert_pcd_to_base(extrinsic_martix=self.extrinsic_matrix_1, pcd=frame_pcds[0])
            pcd_cam2_base = self.convert_pcd_to_base(extrinsic_martix=self.extrinsic_matrix_2, pcd=frame_pcds[1])
            pcd_cam3_base = self.convert_pcd_to_base(extrinsic_martix=self.extrinsic_matrix_3, pcd=frame_pcds[2])

            # Convert to torch tensors
            pcd_cam1_flat = torch.from_numpy(np.ascontiguousarray(pcd_cam1_base)).float().view(-1, 3)
            pcd_cam2_flat = torch.from_numpy(np.ascontiguousarray(pcd_cam2_base)).float().view(-1, 3)
            pcd_cam3_flat = torch.from_numpy(np.ascontiguousarray(pcd_cam3_base)).float().view(-1, 3)

            feat_cam1_flat = ((frame_feats_norm[0].view(-1, 3) + 1) / 2).float()
            feat_cam2_flat = ((frame_feats_norm[1].view(-1, 3) + 1) / 2).float()
            feat_cam3_flat = ((frame_feats_norm[2].view(-1, 3) + 1) / 2).float()

            if self.use_different_projection:
                merged_pcd_list.append([pcd_cam1_flat, pcd_cam2_flat, pcd_cam3_flat])
                merged_feat_list.append([feat_cam1_flat, feat_cam2_flat, feat_cam3_flat])
            else:
                if self.use_merged_pointcloud:
                    merged_pcd = torch.cat([pcd_cam1_flat, pcd_cam2_flat, pcd_cam3_flat], dim=0)
                    merged_feat = torch.cat([feat_cam1_flat, feat_cam2_flat, feat_cam3_flat], dim=0)
                else:
                    merged_pcd = pcd_cam3_flat
                    merged_feat = feat_cam3_flat
                merged_pcd_list.append(merged_pcd)
                merged_feat_list.append(merged_feat)

        pc_list = merged_pcd_list
        img_feat_list = merged_feat_list

        with torch.no_grad():
            action_trans_con = torch.from_numpy(np.array(all_poses)).float()[:, :3]
            quat_wxyz = torch.from_numpy(np.array(all_poses)).float()[:, 3:]
            action_rot_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

            # Apply bounds
            processed_pc_list = []
            processed_feat_list = []

            if self.use_different_projection:
                for frame_pcs, frame_feats in zip(pc_list, img_feat_list):
                    processed_frame_pcs = []
                    processed_frame_feats = []
                    for pc, img_feat in zip(frame_pcs, frame_feats):
                        pc, img_feat = self.move_pc_in_bound(
                            pc.unsqueeze(0), img_feat.unsqueeze(0), self.scene_bounds
                        )
                        processed_frame_pcs.append(pc[0])
                        processed_frame_feats.append(img_feat[0])
                    processed_pc_list.append(processed_frame_pcs)
                    processed_feat_list.append(processed_frame_feats)
            else:
                for pc, img_feat in zip(pc_list, img_feat_list):
                    pc, img_feat = self.move_pc_in_bound(
                        pc.unsqueeze(0), img_feat.unsqueeze(0), self.scene_bounds
                    )
                    processed_pc_list.append(pc[0])
                    processed_feat_list.append(img_feat[0])

            # Place in cube
            if self.use_different_projection:
                first_frame_merged = torch.cat(processed_pc_list[0], dim=0)
                wpt_local, rev_trans = mvt_utils.place_pc_in_cube(
                    first_frame_merged,
                    action_trans_con,
                    with_mean_or_bounds=False,
                    scene_bounds=self.scene_bounds,
                )
            else:
                wpt_local, rev_trans = mvt_utils.place_pc_in_cube(
                    processed_pc_list[0],
                    action_trans_con,
                    with_mean_or_bounds=False,
                    scene_bounds=self.scene_bounds,
                )

            # Normalize point clouds
            final_pc_list = []
            if self.use_different_projection:
                for frame_pcs in processed_pc_list:
                    final_frame_pcs = []
                    for pc in frame_pcs:
                        pc_normalized = mvt_utils.place_pc_in_cube(
                            pc,
                            with_mean_or_bounds=False,
                            scene_bounds=self.scene_bounds,
                        )[0]
                        final_frame_pcs.append(pc_normalized)
                    final_pc_list.append(final_frame_pcs)
            else:
                for pc in processed_pc_list:
                    pc = mvt_utils.place_pc_in_cube(
                        pc,
                        with_mean_or_bounds=False,
                        scene_bounds=self.scene_bounds,
                    )[0]
                    final_pc_list.append(pc)

        return final_pc_list, processed_feat_list, wpt_local, action_rot_xyzw, rev_trans

    @staticmethod
    def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
        """Move point cloud within bounds"""
        if no_op:
            return pc, img_feat

        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        inv_pnt = (
            (pc[:, :, 0] < x_min)
            | (pc[:, :, 0] > x_max)
            | (pc[:, :, 1] < y_min)
            | (pc[:, :, 1] > y_max)
            | (pc[:, :, 2] < z_min)
            | (pc[:, :, 2] > z_max)
            | torch.isnan(pc[:, :, 0])
            | torch.isnan(pc[:, :, 1])
            | torch.isnan(pc[:, :, 2])
        )

        pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
        img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
        return pc, img_feat

    def get_rgb_images(self, processed_pcd, processed_rgb):
        """
        Generate RGB images from processed point cloud

        Args:
            processed_pcd: Processed point cloud (torch.Tensor)
            processed_rgb: Processed RGB features (torch.Tensor)

        Returns:
            List of PIL Images for each view
        """
        rgb_image = self.projection_interface.project_pointcloud_to_rgb(
            processed_pcd, processed_rgb,
            img_aug_before=0.0,
            img_aug_after=0.0
        )  # (1, num_views, H, W, 6)

        rgb_image = rgb_image[0, :, :, :, 3:]  # (num_views, H, W, 3)

        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()

        num_views = rgb_image.shape[0]
        rgb_images = []
        for view_idx in range(num_views):
            view_img = rgb_image[view_idx]
            view_img = (view_img * 255).astype(np.uint8)
            pil_img = Image.fromarray(view_img)
            rgb_images.append(pil_img)

        return rgb_images

    def get_heatmap_images(self, processed_pos):
        """
        Generate heatmap images from processed position

        Args:
            processed_pos: Processed position [num_poses, 3] (torch.Tensor)

        Returns:
            heatmap_images: List of PIL Images for each view
            peak_positions: List of (x, y) tuples for peak position in each view
        """
        img_locations = self.projection_interface.project_pose_to_pixel(
            processed_pos.unsqueeze(0).to(self.projection_interface.renderer_device)
        )  # (1, num_poses, num_views, 2)

        heatmap_sequence = self.projection_interface.generate_heatmap_from_img_locations(
            img_locations,
            self.img_size[0], self.img_size[1],
            self.sigma
        )  # (1, num_poses, num_views, H, W)

        heatmap = heatmap_sequence[0, 0, :, :, :]  # (num_views, H, W)

        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()

        num_views = heatmap.shape[0]
        heatmap_images = []
        peak_positions = []

        for view_idx in range(num_views):
            view_hm = heatmap[view_idx]
            view_hm_min = view_hm.min()
            view_hm_max = view_hm.max()

            # Find peak position (maximum value location)
            peak_idx = np.unravel_index(np.argmax(view_hm), view_hm.shape)
            peak_y, peak_x = peak_idx  # Note: numpy returns (row, col) = (y, x)
            peak_positions.append((peak_x, peak_y))

            if view_hm_max > view_hm_min:
                view_hm_norm = (view_hm - view_hm_min) / (view_hm_max - view_hm_min)
            else:
                view_hm_norm = view_hm

            view_hm_uint8 = (view_hm_norm * 255).astype(np.uint8)
            view_hm_colored = cv2.applyColorMap(view_hm_uint8, cv2.COLORMAP_JET)
            view_hm_colored = cv2.cvtColor(view_hm_colored, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(view_hm_colored)
            heatmap_images.append(pil_img)

        return heatmap_images, peak_positions

    def visualize_all(self, rgb_images, heatmap_images, raw_rgb_images=None,
                      peak_positions=None, title="Debug Visualization", save_path=None):
        """
        Visualize all images in a single figure

        Args:
            rgb_images: List of projected RGB images
            heatmap_images: List of heatmap images
            raw_rgb_images: Optional list of raw camera RGB images
            peak_positions: Optional list of (x, y) tuples for peak positions in each view
            title: Figure title
            save_path: Optional path to save figure
        """
        num_views = len(rgb_images)
        num_rows = 3 if raw_rgb_images else 2

        fig, axes = plt.subplots(num_rows, num_views, figsize=(num_views * 4, num_rows * 4))
        fig.suptitle(title, fontsize=16)

        # Row 1: Projected RGB with peak positions
        for i, img in enumerate(rgb_images):
            ax = axes[0, i] if num_rows > 1 else axes[i]
            ax.imshow(img)
            # Draw peak position marker on RGB image
            if peak_positions and i < len(peak_positions):
                peak_x, peak_y = peak_positions[i]
                # Draw crosshair marker
                ax.scatter(peak_x, peak_y, c='red', s=100, marker='+', linewidths=2, zorder=10)
                # Draw circle around peak
                circle = plt.Circle((peak_x, peak_y), radius=8, fill=False,
                                    color='red', linewidth=2, zorder=10)
                ax.add_patch(circle)
                ax.set_title(f"Projected RGB View {i+1}\nPeak: ({peak_x}, {peak_y})")
            else:
                ax.set_title(f"Projected RGB View {i+1}")
            ax.axis('off')

        # Row 2: Heatmap with peak positions
        for i, img in enumerate(heatmap_images):
            ax = axes[1, i] if num_rows > 1 else axes[i]
            ax.imshow(img)
            # Draw peak position marker on heatmap
            if peak_positions and i < len(peak_positions):
                peak_x, peak_y = peak_positions[i]
                ax.scatter(peak_x, peak_y, c='white', s=100, marker='+', linewidths=2, zorder=10)
                circle = plt.Circle((peak_x, peak_y), radius=8, fill=False,
                                    color='white', linewidth=2, zorder=10)
                ax.add_patch(circle)
                ax.set_title(f"Heatmap View {i+1}\nPeak: ({peak_x}, {peak_y})")
            else:
                ax.set_title(f"Heatmap View {i+1}")
            ax.axis('off')

        # Row 3: Raw RGB (if provided)
        if raw_rgb_images:
            for i, img in enumerate(raw_rgb_images):
                ax = axes[2, i]
                ax.imshow(img)
                ax.set_title(f"Raw Camera {i+1}")
                ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def visualize_pointcloud(self, processed_pcd, processed_rgb=None, title="Point Cloud"):
        """Visualize processed point cloud using Open3D"""
        if isinstance(processed_pcd, list):
            # Merge multiple point clouds
            if isinstance(processed_pcd[0], list):
                processed_pcd = torch.cat([torch.cat(frame_pcs, dim=0) for frame_pcs in processed_pcd], dim=0)
            else:
                processed_pcd = torch.cat(processed_pcd, dim=0)

        if isinstance(processed_pcd, torch.Tensor):
            pcd_np = processed_pcd.cpu().numpy()
        else:
            pcd_np = processed_pcd

        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)

        if processed_rgb is not None:
            if isinstance(processed_rgb, list):
                if isinstance(processed_rgb[0], list):
                    processed_rgb = torch.cat([torch.cat(frame_feats, dim=0) for frame_feats in processed_rgb], dim=0)
                else:
                    processed_rgb = torch.cat(processed_rgb, dim=0)

            if isinstance(processed_rgb, torch.Tensor):
                rgb_np = processed_rgb.cpu().numpy()
            else:
                rgb_np = processed_rgb

            if rgb_np.max() > 1.0:
                rgb_np = rgb_np / 255.0

            pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_np)

        print(f"\n{'='*60}")
        print(f"Visualizing: {title}")
        print(f"Number of points: {len(pcd_o3d.points)}")
        print(f"{'='*60}\n")

        o3d.visualization.draw_geometries(
            [pcd_o3d],
            window_name=title,
            width=800,
            height=600
        )


def capture_and_debug(debugger, camera, robot, save_dir=None):
    """
    Capture images and debug visualization

    Args:
        debugger: ImageDebugger instance
        camera: Camera instance
        robot: Robot controller instance
        save_dir: Optional directory to save images
    """
    from scipy.spatial.transform import Rotation as R

    print("\n" + "="*60)
    print("Capturing observations...")
    print("="*60)

    # Capture from cameras
    obs = camera.capture()

    # Get RGB and point clouds
    rgb_3rd_1 = cv2.cvtColor(obs['3rd_1']['rgb'], cv2.COLOR_BGR2RGB)
    rgb_3rd_2 = cv2.cvtColor(obs['3rd_2']['rgb'], cv2.COLOR_BGR2RGB)
    rgb_3rd_3 = cv2.cvtColor(obs['3rd_3']['rgb'], cv2.COLOR_BGR2RGB)

    pcd_3rd_1 = obs['3rd_1']['pcd']
    pcd_3rd_2 = obs['3rd_2']['pcd']
    pcd_3rd_3 = obs['3rd_3']['pcd']

    # Get current robot pose
    current_pose_obj = robot.get_pose()
    current_position = current_pose_obj.translation
    current_quat = current_pose_obj.quaternion  # [w, x, y, z]

    # Convert to scipy format [x, y, z, w]
    quat_scipy = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
    current_rotation_deg = R.from_quat(quat_scipy).as_euler("xyz", degrees=True)

    print(f"Current position: {current_position}")
    print(f"Current rotation (deg): {current_rotation_deg}")

    # Create pose array
    current_pose = np.concatenate([current_position, current_quat]).reshape(1, 7)

    # Preprocess
    print("\nPreprocessing point clouds...")
    processed_pcd_list, processed_rgb_list, processed_pos, processed_rot, rev_trans = debugger.preprocess(
        [[pcd_3rd_1, pcd_3rd_2, pcd_3rd_3]],
        [[rgb_3rd_1, rgb_3rd_2, rgb_3rd_3]],
        current_pose
    )

    processed_pcd = processed_pcd_list[0]
    processed_rgb = processed_rgb_list[0]

    # Generate images
    print("Generating RGB images...")
    rgb_images = debugger.get_rgb_images(processed_pcd, processed_rgb)

    print("Generating heatmap images...")
    heatmap_images, peak_positions = debugger.get_heatmap_images(processed_pos)
    print(f"Peak positions: {peak_positions}")

    # Create raw RGB images for comparison
    raw_rgb_images = [
        Image.fromarray(cv2.resize(rgb_3rd_1, (256, 256))),
        Image.fromarray(cv2.resize(rgb_3rd_2, (256, 256))),
        Image.fromarray(cv2.resize(rgb_3rd_3, (256, 256)))
    ]

    # Save if requested
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = None
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"debug_{timestamp}.png"

        # Also save individual images
        for i, img in enumerate(rgb_images):
            img.save(save_dir / f"rgb_view{i+1}_{timestamp}.png")
        for i, img in enumerate(heatmap_images):
            img.save(save_dir / f"heatmap_view{i+1}_{timestamp}.png")
        for i, img in enumerate(raw_rgb_images):
            img.save(save_dir / f"raw_cam{i+1}_{timestamp}.png")

    # Visualize
    debugger.visualize_all(
        rgb_images, heatmap_images, raw_rgb_images,
        peak_positions=peak_positions,
        title=f"Debug @ {timestamp}",
        save_path=save_path
    )

    return rgb_images, heatmap_images, processed_pcd, processed_rgb, peak_positions


def main():
    parser = argparse.ArgumentParser(description="Debug RGB and Heatmap image inputs")

    parser.add_argument('--continuous', action='store_true',
                        help='Continuous capture mode (press Enter to capture, q to quit)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save debug images')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--visualize-pcd', action='store_true',
                        help='Also visualize point cloud')
    parser.add_argument('--scene-bounds', type=str, default=None,
                        help='Scene bounds: x_min,y_min,z_min,x_max,y_max,z_max')

    args = parser.parse_args()

    # Parse scene bounds
    scene_bounds = None
    if args.scene_bounds:
        scene_bounds = [float(x) for x in args.scene_bounds.split(',')]

    # Initialize
    print("="*60)
    print("Image Input Debugger")
    print("="*60)

    # Import dependencies
    try:
        import rospy
        from real_camera_utils_lpy import Camera
        from robot_interface import RobotController
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Make sure you're running in the correct environment")
        return

    # Initialize ROS
    if not rospy.core.is_initialized():
        rospy.init_node('image_debugger', anonymous=False, disable_signals=True)

    # Initialize debugger
    debugger = ImageDebugger(
        img_size=args.img_size,
        scene_bounds=scene_bounds
    )

    # Initialize camera
    print("\nInitializing cameras...")
    camera = Camera(camera_type="all")
    print("Cameras initialized")

    # Initialize robot
    print("\nInitializing robot controller...")
    robot = RobotController(frequency=10,with_gripper=False)
    print("Robot controller initialized")

    try:
        if args.continuous:
            print("\n" + "="*60)
            print("Continuous capture mode")
            print("Press Enter to capture, 'q' to quit, 'p' to visualize point cloud")
            print("="*60)

            while True:
                user_input = input("\nPress Enter to capture (q=quit, p=pointcloud): ").strip().lower()

                if user_input == 'q':
                    break
                elif user_input == 'p':
                    # Capture and visualize point cloud
                    rgb_images, heatmap_images, pcd, rgb, peak_pos = capture_and_debug(
                        debugger, camera, robot, args.save_dir
                    )
                    debugger.visualize_pointcloud(pcd, rgb, "Processed Point Cloud")
                else:
                    capture_and_debug(debugger, camera, robot, args.save_dir)
        else:
            # Single capture
            rgb_images, heatmap_images, pcd, rgb, peak_pos = capture_and_debug(
                debugger, camera, robot, args.save_dir
            )

            if args.visualize_pcd:
                debugger.visualize_pointcloud(pcd, rgb, "Processed Point Cloud")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nCleaning up...")
        try:
            robot.cleanup()
        except:
            pass
        print("Done")


if __name__ == "__main__":
    main()
