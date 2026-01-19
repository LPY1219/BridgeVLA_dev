"""
Heatmap与Colormap相互转换工具函数
基于 /share/project/lpy/BridgeVLA/Wan/reconstruct_heatmap/test_heatmap_peak_accuracy.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# Disable flash attention to avoid compatibility issues
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["DISABLE_FLASH_ATTN"] = "1"

try:
    from diffusers import AutoencoderKLWan
except ImportError as e:
    print(f"ImportError: {e}")
    print("Trying to import without flash attention...")
    import sys

    # Try to mock flash_attn to bypass import error
    class MockFlashAttn:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    sys.modules['flash_attn'] = MockFlashAttn()
    sys.modules['flash_attn.flash_attn_interface'] = MockFlashAttn()

    from diffusers import AutoencoderKLWan


def convert_heatmap_to_colormap(heatmap, colormap_name='jet'):
    """
    Convert heatmap to RGB image using matplotlib colormap (optimized)

    Args:
        heatmap: 2D numpy array representing the heatmap
        colormap_name: Name of the colormap (default: 'jet')

    Returns:
        RGB image as numpy array with shape (H, W, 3) and dtype float32
    """
    # Normalize heatmap to [0, 1] if needed
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Apply colormap (vectorized operation)
    colormap = cm.get_cmap(colormap_name)
    rgb_image = colormap(heatmap_norm)[:, :, :3]  # Remove alpha channel

    return rgb_image.astype(np.float32)


def convert_heatmap_sequence_to_colormap(heatmap_sequence, colormap_name='jet'):
    """
    Convert a sequence of heatmaps to colormap video

    Args:
        heatmap_sequence: numpy array with shape (T, H, W)
        colormap_name: Name of the colormap (default: 'jet')

    Returns:
        RGB video as numpy array with shape (T, H, W, 3)
    """
    T, H, W = heatmap_sequence.shape
    colormap_video = np.zeros((T, H, W, 3), dtype=np.float32)

    for t in range(T):
        colormap_video[t] = convert_heatmap_to_colormap(heatmap_sequence[t], colormap_name)

    return colormap_video


def extract_heatmap_from_colormap(rgb_image, colormap_name='jet'):
    """
    Extract heatmap from RGB colormap image by finding closest colormap values (adaptive)

    Args:
        rgb_image: RGB image with shape (H, W, 3)
        colormap_name: Name of the colormap used (default: 'jet')

    Returns:
        Extracted heatmap as numpy array with shape (H, W) and dtype float32
    """
    h, w = rgb_image.shape[:2]
    colormap = cm.get_cmap(colormap_name)

    # Adaptive algorithm selection based on image size
    total_pixels = h * w
    use_vectorized = total_pixels > 16384  # 128x128 threshold

    if use_vectorized:
        # Vectorized method for large images - optimized to 128 points for speed
        reference_values = np.linspace(0, 1, 128)
        reference_colors = colormap(reference_values)[:, :3]

        rgb_flat = rgb_image.reshape(-1, 3)
        distances = np.sum((rgb_flat[:, None, :] - reference_colors[None, :, :]) ** 2, axis=2)
        closest_indices = np.argmin(distances, axis=1)
        extracted_values = reference_values[closest_indices]
        extracted_heatmap = extracted_values.reshape(h, w)

    else:
        # Loop method for small images - also optimized to 128 points
        reference_values = np.linspace(0, 1, 128)
        reference_colors = colormap(reference_values)[:, :3]

        extracted_heatmap = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                pixel = rgb_image[i, j]
                distances = np.sum((reference_colors - pixel) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                extracted_heatmap[i, j] = reference_values[closest_idx]

    return extracted_heatmap.astype(np.float32)


def extract_heatmap_sequence_from_colormap(colormap_video, colormap_name='jet'):
    """
    Extract heatmap sequence from colormap video

    Args:
        colormap_video: RGB video with shape (T, H, W, 3)
        colormap_name: Name of the colormap used (default: 'jet')

    Returns:
        Heatmap sequence as numpy array with shape (T, H, W)
    """
    # Video case: (T, H, W, 3) -> (T, H, W)
    T, H, W, _ = colormap_video.shape
    heatmap_sequence = np.zeros((T, H, W), dtype=np.float32)

    for t in range(T):
        heatmap_sequence[t] = extract_heatmap_from_colormap(colormap_video[t], colormap_name)

    return heatmap_sequence


def convert_color_to_wan_format(image):
    """
    Convert color image to Wan-VAE format

    Args:
        image: RGB image with shape (H, W, 3)

    Returns:
        Tensor with shape (1, 3, 1, H, W) for Wan VAE input
    """
    # Convert to torch tensor and permute dimensions
    # Shape: (H, W, 3) -> (3, H, W) -> (1, 3, 1, H, W)
    image_tensor = torch.from_numpy(image).float()
    image_chw = image_tensor.permute(2, 0, 1)  # H,W,C -> C,H,W
    image_5d = image_chw.unsqueeze(0).unsqueeze(2)  # Add batch and time dimensions

    return image_5d


def convert_colormap_video_to_wan_format(colormap_video):
    """
    Convert colormap video to Wan-VAE format

    Args:
        colormap_video: RGB video with shape (T, H, W, 3)

    Returns:
        Tensor with shape (1, 3, T, H, W) for Wan VAE input
    """
    # Convert to torch tensor and permute dimensions
    # Shape: (T, H, W, 3) -> (3, T, H, W) -> (1, 3, T, H, W)
    video_tensor = torch.from_numpy(colormap_video).float()
    video_cthw = video_tensor.permute(3, 0, 1, 2)  # T,H,W,C -> C,T,H,W
    video_5d = video_cthw.unsqueeze(0)  # Add batch dimension

    return video_5d


def convert_from_wan_format(decoded_5d):
    """
    Convert decoded output back to image format

    Args:
        decoded_5d: Tensor with shape (1, 3, T, H, W)

    Returns:
        RGB video as numpy array with shape (T, H, W, 3)
        Always returns video format to maintain time dimension consistency
    """
    # Remove batch dimension
    decoded_4d = decoded_5d.squeeze(0)  # (3, T, H, W)

    # Always treat as video to maintain time dimension consistency
    # Video case: (3, T, H, W) -> (T, H, W, 3)
    decoded_hwc = decoded_4d.permute(1, 2, 3, 0).cpu().numpy()  # C,T,H,W -> T,H,W,C

    # Clamp to [0, 1] range
    decoded_hwc = np.clip(decoded_hwc, 0, 1)

    return decoded_hwc


def load_vae_model(model_path="/share/project/lpy/huggingface/Wan_2_2_TI2V_5B_Diffusers"):
    """
    Load WAN-VAE model

    Args:
        model_path: Path to the Wan2.2 model

    Returns:
        Tuple of (vae_model, device)
    """
    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=torch.float32
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = vae.to(device)
    vae.eval()  # Set to evaluation mode

    return vae, device


def encode_decode_heatmap(vae, heatmap_video, device):
    """
    Encode and then decode the heatmap using Wan-VAE

    Args:
        vae: Loaded VAE model
        heatmap_video: Input tensor in Wan format
        device: Device to run on

    Returns:
        Decoded tensor in Wan format
    """
    heatmap_video = heatmap_video.to(device)

    with torch.no_grad():
        # Encode to latent space
        latent_dist = vae.encode(heatmap_video)
        latent = latent_dist.latent_dist.sample()

        # Decode back to image space
        decoded = vae.decode(latent).sample

    return decoded


def test_colormap_conversion():
    """
    Test function to verify colormap conversion accuracy
    """
    # Generate test heatmap
    h, w = 64, 64
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    test_heatmap = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

    # Convert to colormap and back
    colormap = convert_heatmap_to_colormap(test_heatmap)
    recovered_heatmap = extract_heatmap_from_colormap(colormap)

    # Calculate error
    mse = np.mean((test_heatmap - recovered_heatmap) ** 2)
    print(f"Colormap conversion MSE: {mse:.6f}")

    return mse < 0.01  # Should be very small error


if __name__ == "__main__":
    # Run test
    print("Testing colormap conversion...")
    success = test_colormap_conversion()
    print(f"Test {'PASSED' if success else 'FAILED'}")