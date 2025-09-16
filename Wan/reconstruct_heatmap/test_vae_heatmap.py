import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from PIL import Image
import urllib.request
from matplotlib import cm

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


def generate_gaussian_heatmap(height=256, width=256, center_x=128, center_y=128, sigma=50):
    """
    Generate a 2D Gaussian heatmap with normalized probabilities (sum=1)

    Args:
        height: Height of the heatmap
        width: Width of the heatmap
        center_x: X coordinate of the Gaussian center
        center_y: Y coordinate of the Gaussian center
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        2D numpy array representing the normalized Gaussian heatmap
    """
    y, x = np.ogrid[:height, :width]

    # Calculate the 2D Gaussian distribution
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    # Normalize so that the sum of all probabilities equals 1
    gaussian_normalized = gaussian / np.sum(gaussian)

    return gaussian_normalized


def convert_to_wan_format(heatmap):
    """
    Convert single-channel heatmap to Wan-VAE format
    Wan-VAE expects video input: B×C×T×H×W format
    For single frame (T=1), input should be (1, 3, 1, H, W)

    Args:
        heatmap: 2D numpy array

    Returns:
        torch tensor with shape (1, 3, 1, H, W) for Wan-VAE
    """
    # Convert to torch tensor
    heatmap_tensor = torch.from_numpy(heatmap).float()

    # Create 5D tensor: (H, W) -> (1, 3, 1, H, W)
    # Shape: (H, W) -> (1, 1, H, W) -> (1, 3, 1, H, W)
    heatmap_4d = heatmap_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and time
    heatmap_5d = heatmap_4d.repeat(1, 3, 1, 1, 1)  # Repeat across channel dimension

    return heatmap_5d


def convert_heatmap_to_colormap(heatmap, colormap_name='viridis'):
    """
    Convert heatmap to RGB image using matplotlib colormap

    Args:
        heatmap: 2D numpy array with normalized values [0, 1]
        colormap_name: Name of matplotlib colormap

    Returns:
        RGB image with shape (H, W, 3)
    """
    # Normalize heatmap to [0, 1] if needed
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Apply colormap
    colormap = cm.get_cmap(colormap_name)
    rgb_image = colormap(heatmap_norm)[:, :, :3]  # Remove alpha channel

    return rgb_image.astype(np.float32)


def extract_heatmap_from_colormap(rgb_image, colormap_name='viridis'):
    """
    Extract heatmap from RGB colormap image by finding closest colormap values (optimized)

    Args:
        rgb_image: RGB image with shape (H, W, 3)
        colormap_name: Name of matplotlib colormap used

    Returns:
        Extracted heatmap with shape (H, W)
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


def load_color_image(image_path=None, size=(256, 256)):
    """
    Load and preprocess a color image for Wan-VAE testing

    Args:
        image_path: Path to image file, if None, creates a test image
        size: Resize dimensions (H, W)

    Returns:
        numpy array with shape (H, W, 3) normalized to [0, 1]
    """
    if image_path is None:
        # Create a test colorful image
        h, w = size
        image = np.zeros((h, w, 3))

        # Create colorful gradient patterns
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)

        # Red channel: diagonal gradient
        image[:, :, 0] = (X + Y) / 2

        # Green channel: circular pattern
        center_x, center_y = w // 2, h // 2
        image[:, :, 1] = np.exp(-((np.arange(w)[None, :] - center_x)**2 +
                                 (np.arange(h)[:, None] - center_y)**2) / (2 * 50**2))

        # Blue channel: wave pattern
        image[:, :, 2] = 0.5 * (1 + np.sin(X * 10) * np.cos(Y * 10))

    else:
        # Load real image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size)
        image = np.array(img) / 255.0

    return image.astype(np.float32)


def convert_color_to_wan_format(image):
    """
    Convert color image to Wan-VAE format
    Wan-VAE expects video input: B×C×T×H×W format
    For single frame (T=1), input should be (1, 3, 1, H, W)

    Args:
        image: numpy array with shape (H, W, 3)

    Returns:
        torch tensor with shape (1, 3, 1, H, W) for Wan-VAE
    """
    # Convert to torch tensor and permute dimensions
    # Shape: (H, W, 3) -> (3, H, W) -> (1, 3, 1, H, W)
    image_tensor = torch.from_numpy(image).float()
    image_chw = image_tensor.permute(2, 0, 1)  # H,W,C -> C,H,W
    image_5d = image_chw.unsqueeze(0).unsqueeze(2)  # Add batch and time dimensions

    return image_5d


def convert_from_wan_format(decoded_5d):
    """
    Convert decoded output back to image format

    Args:
        decoded_5d: Video tensor with shape (1, 3, 1, H, W)

    Returns:
        numpy array with shape (H, W, 3)
    """
    # Shape: (1, 3, 1, H, W) -> (3, H, W) -> (H, W, 3)
    decoded_chw = decoded_5d.squeeze(0).squeeze(1)  # Remove batch and time dims
    decoded_hwc = decoded_chw.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C

    # Clamp to [0, 1] range
    decoded_hwc = np.clip(decoded_hwc, 0, 1)

    return decoded_hwc


def load_vae_model(model_path="/share/project/lpy/huggingface/Wan_2_2_TI2V_5B_Diffusers"):
    """
    Load WAN2.2 VAE model

    Args:
        model_path: Path to the pre-trained model

    Returns:
        Loaded VAE model
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
    Encode and then decode the heatmap video using Wan-VAE

    Args:
        vae: The Wan-VAE model
        heatmap_video: Video tensor with shape (1, 3, 1, H, W)
        device: Device to run on

    Returns:
        Decoded video tensor
    """
    heatmap_video = heatmap_video.to(device)

    with torch.no_grad():
        # Encode to latent space - Wan-VAE compresses spatially and temporally
        latent_dist = vae.encode(heatmap_video)
        latent = latent_dist.latent_dist.sample()

        # Decode back to video space
        decoded = vae.decode(latent).sample

    return decoded


def convert_to_single_channel(decoded_video):
    """
    Convert decoded video output back to single channel by averaging
    Wan-VAE outputs video tensor: (1, 3, 1, H, W)

    Args:
        decoded_video: Video tensor with shape (1, 3, 1, H, W)

    Returns:
        Single-channel numpy array with shape (H, W)
    """
    # Average across channel dimension and remove batch/time dimensions
    # Shape: (1, 3, 1, H, W) -> (1, 1, H, W) -> (H, W)
    decoded_1ch = torch.mean(decoded_video, dim=1).squeeze(0).squeeze(0).cpu().numpy()

    return decoded_1ch


def visualize_heatmaps(original_heatmap, decoded_heatmap, save_path="heatmap_comparison.png"):
    """
    Visualize original and decoded heatmaps side by side

    Args:
        original_heatmap: Original 2D numpy array
        decoded_heatmap: Decoded 2D numpy array
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original heatmap
    im1 = axes[0].imshow(original_heatmap, cmap='hot', interpolation='nearest')
    axes[0].set_title('Original Heatmap')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # Decoded heatmap
    im2 = axes[1].imshow(decoded_heatmap, cmap='hot', interpolation='nearest')
    axes[1].set_title('Decoded Heatmap')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    # Difference map
    diff = np.abs(original_heatmap - decoded_heatmap)
    im3 = axes[2].imshow(diff, cmap='viridis', interpolation='nearest')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualization saved to: {save_path}")


def visualize_colormap_reconstruction(original_heatmap, decoded_heatmap,
                                   original_colormap, decoded_colormap,
                                   save_path="colormap_reconstruction.png"):
    """
    Visualize complete colormap-based heatmap reconstruction pipeline

    Args:
        original_heatmap: Original heatmap (H, W)
        decoded_heatmap: Decoded heatmap (H, W)
        original_colormap: Original RGB colormap image (H, W, 3)
        decoded_colormap: Decoded RGB colormap image (H, W, 3)
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Heatmaps
    # Original heatmap
    im1 = axes[0, 0].imshow(original_heatmap, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Original Heatmap')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])

    # Original colormap
    axes[0, 1].imshow(original_colormap)
    axes[0, 1].set_title('Original Colormap')
    axes[0, 1].axis('off')

    # Decoded colormap
    axes[0, 2].imshow(decoded_colormap)
    axes[0, 2].set_title('Decoded Colormap')
    axes[0, 2].axis('off')

    # Decoded heatmap
    im4 = axes[0, 3].imshow(decoded_heatmap, cmap='viridis', interpolation='nearest')
    axes[0, 3].set_title('Decoded Heatmap')
    axes[0, 3].axis('off')
    plt.colorbar(im4, ax=axes[0, 3])

    # Row 2: Differences
    # Heatmap difference
    heatmap_diff = np.abs(original_heatmap - decoded_heatmap)
    im5 = axes[1, 0].imshow(heatmap_diff, cmap='Reds', interpolation='nearest')
    axes[1, 0].set_title('Heatmap Difference')
    axes[1, 0].axis('off')
    plt.colorbar(im5, ax=axes[1, 0])

    # Colormap RGB difference
    colormap_diff = np.abs(original_colormap - decoded_colormap)
    im6 = axes[1, 1].imshow(colormap_diff)
    axes[1, 1].set_title('Colormap RGB Difference')
    axes[1, 1].axis('off')

    # Combined difference visualization
    combined_diff = np.mean(colormap_diff, axis=2)
    im7 = axes[1, 2].imshow(combined_diff, cmap='viridis', interpolation='nearest')
    axes[1, 2].set_title('Combined Difference')
    axes[1, 2].axis('off')
    plt.colorbar(im7, ax=axes[1, 2])

    # Statistics text
    axes[1, 3].axis('off')
    stats_text = f"""Reconstruction Quality:

Heatmap MSE: {np.mean(heatmap_diff**2):.6f}
Heatmap MAE: {np.mean(heatmap_diff):.6f}

Colormap MSE: {np.mean(colormap_diff**2):.6f}
Colormap MAE: {np.mean(colormap_diff):.6f}

Max Heatmap Diff: {np.max(heatmap_diff):.6f}
Max Colormap Diff: {np.max(colormap_diff):.6f}
    """
    axes[1, 3].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Colormap reconstruction visualization saved to: {save_path}")


def visualize_color_images(original_image, decoded_image, save_path="color_comparison.png"):
    """
    Visualize original and decoded color images side by side

    Args:
        original_image: Original image with shape (H, W, 3)
        decoded_image: Decoded image with shape (H, W, 3)
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Decoded image
    axes[1].imshow(decoded_image)
    axes[1].set_title('Decoded Image')
    axes[1].axis('off')

    # Difference map
    diff = np.abs(original_image - decoded_image)
    im3 = axes[2].imshow(diff, cmap='viridis', interpolation='nearest')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Color image visualization saved to: {save_path}")


def calculate_color_metrics(original, decoded):
    """
    Calculate reconstruction metrics for color images

    Args:
        original: Original color image
        decoded: Decoded color image

    Returns:
        Dictionary of metrics
    """
    mse = np.mean((original - decoded) ** 2)
    mae = np.mean(np.abs(original - decoded))

    # Correlation coefficient (computed across all pixels and channels)
    orig_flat = original.flatten()
    dec_flat = decoded.flatten()
    corr = np.corrcoef(orig_flat, dec_flat)[0, 1]

    # Peak signal-to-noise ratio
    max_val = 1.0  # Since images are normalized to [0, 1]
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')

    return {
        'MSE': mse,
        'MAE': mae,
        'Correlation': corr,
        'PSNR': psnr
    }


def calculate_metrics(original, decoded):
    """
    Calculate reconstruction metrics for heatmaps

    Args:
        original: Original heatmap
        decoded: Decoded heatmap

    Returns:
        Dictionary of metrics
    """
    mse = np.mean((original - decoded) ** 2)
    mae = np.mean(np.abs(original - decoded))

    # Correlation coefficient
    corr = np.corrcoef(original.flatten(), decoded.flatten())[0, 1]

    # Peak signal-to-noise ratio
    max_val = np.max(original)
    psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else float('inf')

    return {
        'MSE': mse,
        'MAE': mae,
        'Correlation': corr,
        'PSNR': psnr
    }


def calculate_heatmap_distribution_metrics(original, decoded):
    """
    Calculate distribution-specific metrics for heatmaps (KL divergence, etc.)

    Args:
        original: Original heatmap (normalized probability distribution)
        decoded: Decoded heatmap

    Returns:
        Dictionary of distribution metrics
    """
    # Normalize both to probability distributions
    original_norm = original / (np.sum(original) + 1e-8)
    decoded_norm = decoded / (np.sum(decoded) + 1e-8)

    # Add small epsilon to avoid log(0)
    eps = 1e-8
    original_safe = original_norm + eps
    decoded_safe = decoded_norm + eps

    # KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
    kl_divergence = np.sum(original_safe * np.log(original_safe / decoded_safe))

    # JS divergence: symmetric version of KL
    m = 0.5 * (original_safe + decoded_safe)
    js_divergence = 0.5 * np.sum(original_safe * np.log(original_safe / m)) + \
                   0.5 * np.sum(decoded_safe * np.log(decoded_safe / m))

    # Earth Mover's Distance (simplified 1D version)
    # Flatten and compute cumulative distributions
    orig_flat = original_norm.flatten()
    dec_flat = decoded_norm.flatten()
    orig_cumsum = np.cumsum(orig_flat)
    dec_cumsum = np.cumsum(dec_flat)
    emd = np.mean(np.abs(orig_cumsum - dec_cumsum))

    # Total Variation distance
    tv_distance = 0.5 * np.sum(np.abs(original_norm - decoded_norm))

    return {
        'KL_divergence': kl_divergence,
        'JS_divergence': js_divergence,
        'EMD': emd,
        'TV_distance': tv_distance
    }


def test_color_image_reconstruction(vae, device):
    """
    Test color image reconstruction with Wan-VAE
    """
    print("\n" + "="*60)
    print("Testing Wan-VAE color image reconstruction...")

    # Step 1: Load color image
    print("Step 1: Loading test color image...")
    original_image = load_color_image()
    print(f"Original image shape: {original_image.shape}")
    print(f"Image value range: [{original_image.min():.3f}, {original_image.max():.3f}]")

    # Step 2: Convert to Wan-VAE format
    print("Step 2: Converting to Wan-VAE format...")
    image_5d = convert_color_to_wan_format(original_image)
    print(f"5D tensor shape: {image_5d.shape}")

    # Step 3: Encode and decode
    print("Step 3: Encoding and decoding with Wan-VAE...")
    decoded_5d = encode_decode_heatmap(vae, image_5d, device)
    print(f"Decoded 5D shape: {decoded_5d.shape}")

    # Step 4: Convert back to image format
    print("Step 4: Converting back to image format...")
    decoded_image = convert_from_wan_format(decoded_5d)
    print(f"Decoded image shape: {decoded_image.shape}")
    print(f"Decoded value range: [{decoded_image.min():.3f}, {decoded_image.max():.3f}]")

    # Step 5: Calculate metrics
    print("Step 5: Calculating reconstruction metrics...")
    metrics = calculate_color_metrics(original_image, decoded_image)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    # Step 6: Visualize results
    print("Step 6: Visualizing color reconstruction...")
    visualize_color_images(original_image, decoded_image)

    return original_image, decoded_image


def test_colormap_heatmap_reconstruction(vae, device, colormap_name='viridis'):
    """
    Test heatmap reconstruction using colormap conversion approach
    """
    print("\n" + "="*60)
    print(f"Testing colormap-based heatmap reconstruction (colormap: {colormap_name})...")

    # Step 1: Generate original heatmap
    print("Step 1: Generating original heatmap...")
    original_heatmap = generate_gaussian_heatmap(256, 256, center_x=128, center_y=128, sigma=40)
    print(f"Original heatmap shape: {original_heatmap.shape}")
    print(f"Original heatmap sum: {np.sum(original_heatmap):.6f}")

    # Step 2: Convert heatmap to colormap RGB
    print("Step 2: Converting heatmap to colormap RGB...")
    original_colormap = convert_heatmap_to_colormap(original_heatmap, colormap_name)
    print(f"Original colormap shape: {original_colormap.shape}")
    print(f"Colormap value range: [{original_colormap.min():.3f}, {original_colormap.max():.3f}]")

    # Step 3: Convert to Wan-VAE format
    print("Step 3: Converting to Wan-VAE format...")
    colormap_5d = convert_color_to_wan_format(original_colormap)
    print(f"5D tensor shape: {colormap_5d.shape}")

    # Step 4: Encode and decode with VAE
    print("Step 4: Encoding and decoding with Wan-VAE...")
    decoded_5d = encode_decode_heatmap(vae, colormap_5d, device)
    print(f"Decoded 5D shape: {decoded_5d.shape}")

    # Step 5: Convert back to RGB colormap
    print("Step 5: Converting back to RGB colormap...")
    decoded_colormap = convert_from_wan_format(decoded_5d)
    print(f"Decoded colormap shape: {decoded_colormap.shape}")
    print(f"Decoded colormap range: [{decoded_colormap.min():.3f}, {decoded_colormap.max():.3f}]")

    # Step 6: Extract heatmap from decoded colormap
    print("Step 6: Extracting heatmap from decoded colormap...")
    decoded_heatmap = extract_heatmap_from_colormap(decoded_colormap, colormap_name)
    print(f"Decoded heatmap shape: {decoded_heatmap.shape}")
    print(f"Decoded heatmap sum: {np.sum(decoded_heatmap):.6f}")

    # Step 7: Calculate standard metrics
    print("Step 7: Calculating standard reconstruction metrics...")
    standard_metrics = calculate_metrics(original_heatmap, decoded_heatmap)
    print("Standard metrics:")
    for metric, value in standard_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Step 8: Calculate distribution-specific metrics
    print("Step 8: Calculating distribution-specific metrics...")
    dist_metrics = calculate_heatmap_distribution_metrics(original_heatmap, decoded_heatmap)
    print("Distribution metrics:")
    for metric, value in dist_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Step 9: Calculate colormap reconstruction metrics
    print("Step 9: Calculating colormap reconstruction metrics...")
    colormap_metrics = calculate_color_metrics(original_colormap, decoded_colormap)
    print("Colormap metrics:")
    for metric, value in colormap_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Step 10: Visualize complete reconstruction pipeline
    print("Step 10: Visualizing complete reconstruction...")
    visualize_colormap_reconstruction(
        original_heatmap, decoded_heatmap,
        original_colormap, decoded_colormap,
        save_path=f"colormap_reconstruction_{colormap_name}.png"
    )

    return original_heatmap, decoded_heatmap, original_colormap, decoded_colormap


def main():
    """
    Main function to test Wan-VAE zero-shot encoding/decoding
    """
    print("Testing Wan-VAE zero-shot reconstruction...")

    # Load VAE model (shared for both tests)
    print("Loading Wan-VAE model...")
    vae, device = load_vae_model()
    print(f"VAE loaded on device: {device}")

    # Test 1: Heatmap reconstruction
    print("\n" + "="*60)
    print("Testing heatmap reconstruction...")

    # Step 1: Generate Gaussian heatmap
    print("Step 1: Generating 256x256 Gaussian heatmap...")
    original_heatmap = generate_gaussian_heatmap(256, 256, center_x=128, center_y=128, sigma=40)
    print(f"Original heatmap shape: {original_heatmap.shape}")
    print(f"Original heatmap sum: {np.sum(original_heatmap):.6f}")

    # Step 2: Convert to Wan-VAE video format
    print("Step 2: Converting to Wan-VAE video format...")
    heatmap_video = convert_to_wan_format(original_heatmap)
    print(f"Video heatmap shape: {heatmap_video.shape}")

    # Step 3: Encode and decode
    print("Step 3: Encoding and decoding with Wan-VAE...")
    decoded_video = encode_decode_heatmap(vae, heatmap_video, device)
    print(f"Decoded video shape: {decoded_video.shape}")

    # Step 4: Convert back to single channel
    print("Step 4: Converting back to single channel...")
    decoded_heatmap = convert_to_single_channel(decoded_video)
    print(f"Decoded heatmap shape: {decoded_heatmap.shape}")

    # Step 5: Calculate metrics
    print("Step 5: Calculating reconstruction metrics...")
    metrics = calculate_metrics(original_heatmap, decoded_heatmap)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    # Step 6: Visualize results
    print("Step 6: Visualizing heatmap results...")
    visualize_heatmaps(original_heatmap, decoded_heatmap)

    # Test 2: Color image reconstruction
    original_color, decoded_color = test_color_image_reconstruction(vae, device)

    # Test 3: Colormap-based heatmap reconstruction
    original_hm, decoded_hm, original_cm, decoded_cm = test_colormap_heatmap_reconstruction(vae, device, 'viridis')

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("\nSummary:")
    print("1. Standard heatmap reconstruction (channel repetition)")
    print("2. Color image reconstruction (natural RGB)")
    print("3. Colormap-based heatmap reconstruction (heatmap -> RGB -> heatmap)")


if __name__ == "__main__":
    main()