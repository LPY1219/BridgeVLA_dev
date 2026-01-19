import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
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


def generate_gaussian_heatmap(height=256, width=256, center_x=None, center_y=None, sigma=5):
    """
    Generate a 2D Gaussian heatmap with normalized probabilities (sum=1)

    Args:
        height: Height of the heatmap
        width: Width of the heatmap
        center_x: X coordinate of the Gaussian center (random if None)
        center_y: Y coordinate of the Gaussian center (random if None)
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        2D numpy array representing the normalized Gaussian heatmap
    """
    if center_x is None:
        center_x = random.randint(sigma*3, width - sigma*3)
    if center_y is None:
        center_y = random.randint(sigma*3, height - sigma*3)

    y, x = np.ogrid[:height, :width]

    # Calculate the 2D Gaussian distribution
    gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    # Normalize so that the sum of all probabilities equals 1
    gaussian_normalized = gaussian / np.sum(gaussian)

    return gaussian_normalized, (center_x, center_y)


def convert_heatmap_to_colormap(heatmap, colormap_name='jet'):
    """
    Convert heatmap to RGB image using matplotlib colormap (optimized)
    """
    # Normalize heatmap to [0, 1] if needed
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Apply colormap (vectorized operation)
    colormap = cm.get_cmap(colormap_name)
    rgb_image = colormap(heatmap_norm)[:, :, :3]  # Remove alpha channel

    return rgb_image.astype(np.float32)


def extract_heatmap_from_colormap(rgb_image, colormap_name='jet'):
    """
    Extract heatmap from RGB colormap image by finding closest colormap values (adaptive)
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


def convert_color_to_wan_format(image):
    """
    Convert color image to Wan-VAE format
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
    """
    # Shape: (1, 3, 1, H, W) -> (3, H, W) -> (H, W, 3)
    decoded_chw = decoded_5d.squeeze(0).squeeze(1)  # Remove batch and time dims
    decoded_hwc = decoded_chw.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C

    # Clamp to [0, 1] range
    decoded_hwc = np.clip(decoded_hwc, 0, 1)

    return decoded_hwc


def load_vae_model(model_path="/share/project/lpy/huggingface/Wan_2_2_TI2V_5B_Diffusers"):
    """
    Load WAN-VAE model
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
    """
    heatmap_video = heatmap_video.to(device)

    with torch.no_grad():
        # Encode to latent space
        latent_dist = vae.encode(heatmap_video)
        latent = latent_dist.latent_dist.sample()

        # Decode back to image space
        decoded = vae.decode(latent).sample

    return decoded


def find_peak_location(heatmap):
    """
    Find the location of maximum value in heatmap

    Returns:
        (x, y) coordinates of the peak
    """
    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (peak_idx[1], peak_idx[0])  # Return as (x, y)


def calculate_peak_distance(original_peak, decoded_peak):
    """
    Calculate Euclidean distance between two peak locations
    """
    return np.sqrt((original_peak[0] - decoded_peak[0])**2 +
                   (original_peak[1] - decoded_peak[1])**2)


def visualize_heatmap_comparison(original_heatmap, decoded_heatmap,
                               original_peak, decoded_peak,
                               test_id, save_path):
    """
    Visualize original and decoded heatmaps with peak locations marked
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original heatmap
    im1 = axes[0].imshow(original_heatmap, cmap='jet', interpolation='nearest')
    axes[0].scatter(original_peak[0], original_peak[1], c='red', s=100, marker='x', linewidths=3)
    axes[0].set_title(f'Original Heatmap (Test {test_id})\nPeak: ({original_peak[0]}, {original_peak[1]})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # Decoded heatmap
    im2 = axes[1].imshow(decoded_heatmap, cmap='jet', interpolation='nearest')
    axes[1].scatter(decoded_peak[0], decoded_peak[1], c='red', s=100, marker='x', linewidths=3)
    axes[1].set_title(f'Decoded Heatmap (Test {test_id})\nPeak: ({decoded_peak[0]}, {decoded_peak[1]})')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    # Difference map
    diff = np.abs(original_heatmap - decoded_heatmap)
    im3 = axes[2].imshow(diff, cmap='Reds', interpolation='nearest')

    # Mark both peaks on difference map
    axes[2].scatter(original_peak[0], original_peak[1], c='blue', s=100, marker='o',
                   linewidths=2, label='Original Peak')
    axes[2].scatter(decoded_peak[0], decoded_peak[1], c='red', s=100, marker='x',
                   linewidths=3, label='Decoded Peak')

    # Draw line between peaks
    axes[2].plot([original_peak[0], decoded_peak[0]],
                [original_peak[1], decoded_peak[1]],
                'yellow', linewidth=2, linestyle='--')

    distance = calculate_peak_distance(original_peak, decoded_peak)
    axes[2].set_title(f'Absolute Difference\nPeak Distance: {distance:.2f} pixels')
    axes[2].axis('off')
    axes[2].legend()
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualization for test {test_id} saved to: {save_path}")


def run_peak_accuracy_tests(num_tests=20, sigma=5):
    """
    Run multiple tests to evaluate peak location accuracy
    """
    print(f"Running {num_tests} peak accuracy tests with sigma={sigma}...")
    print("="*60)

    # Load VAE model
    print("Loading Wan-VAE model...")
    vae, device = load_vae_model()
    print(f"VAE loaded on device: {device}")

    # Storage for results
    peak_distances = []
    test_results = []

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    for test_id in range(1, num_tests + 1):
        print(f"\nTest {test_id}/{num_tests}:")
        print("-" * 30)

        # Step 1: Generate random Gaussian heatmap
        original_heatmap, true_center = generate_gaussian_heatmap(
            height=256, width=256, sigma=sigma
        )
        original_peak = find_peak_location(original_heatmap)

        print(f"Generated heatmap with center: {true_center}, detected peak: {original_peak}")

        # Step 2: Convert to colormap and process through VAE
        original_colormap = convert_heatmap_to_colormap(original_heatmap, 'jet')
        colormap_5d = convert_color_to_wan_format(original_colormap)

        # Step 3: Encode and decode
        decoded_5d = encode_decode_heatmap(vae, colormap_5d, device)
        decoded_colormap = convert_from_wan_format(decoded_5d)

        # Step 4: Extract heatmap from decoded colormap
        decoded_heatmap = extract_heatmap_from_colormap(decoded_colormap, 'jet')
        decoded_peak = find_peak_location(decoded_heatmap)

        # Step 5: Calculate peak distance
        distance = calculate_peak_distance(original_peak, decoded_peak)
        peak_distances.append(distance)

        print(f"Decoded peak: {decoded_peak}")
        print(f"Peak distance: {distance:.2f} pixels")

        # Store test results
        test_results.append({
            'test_id': test_id,
            'true_center': true_center,
            'original_peak': original_peak,
            'decoded_peak': decoded_peak,
            'distance': distance,
            'original_heatmap': original_heatmap,
            'decoded_heatmap': decoded_heatmap
        })

    # Calculate statistics
    mean_distance = np.mean(peak_distances)
    std_distance = np.std(peak_distances)
    min_distance = np.min(peak_distances)
    max_distance = np.max(peak_distances)

    print("\n" + "="*60)
    print("PEAK ACCURACY TEST RESULTS")
    print("="*60)
    print(f"Number of tests: {num_tests}")
    print(f"Gaussian sigma: {sigma}")
    print(f"Image size: 256x256")
    print()
    print("Peak Distance Statistics:")
    print(f"  Mean distance: {mean_distance:.2f} ± {std_distance:.2f} pixels")
    print(f"  Min distance:  {min_distance:.2f} pixels")
    print(f"  Max distance:  {max_distance:.2f} pixels")
    print()

    # Find best and worst cases
    best_idx = np.argmin(peak_distances)
    worst_idx = np.argmax(peak_distances)

    print(f"Best case (Test {best_idx + 1}): {peak_distances[best_idx]:.2f} pixels")
    print(f"Worst case (Test {worst_idx + 1}): {peak_distances[worst_idx]:.2f} pixels")

    # Randomly select 2 tests for visualization
    random.seed(42)
    viz_indices = random.sample(range(num_tests), 2)

    print(f"\nRandomly selected tests for visualization: {[i+1 for i in viz_indices]}")

    for i, viz_idx in enumerate(viz_indices):
        result = test_results[viz_idx]
        visualize_heatmap_comparison(
            result['original_heatmap'],
            result['decoded_heatmap'],
            result['original_peak'],
            result['decoded_peak'],
            result['test_id'],
            f"peak_accuracy_test_{result['test_id']}.png"
        )

    # Create summary plot
    plt.figure(figsize=(12, 8))

    # Subplot 1: Distance distribution
    plt.subplot(2, 2, 1)
    plt.hist(peak_distances, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_distance, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_distance:.2f}')
    plt.xlabel('Peak Distance (pixels)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Peak Distances')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Distance vs test number
    plt.subplot(2, 2, 2)
    test_numbers = list(range(1, num_tests + 1))
    plt.plot(test_numbers, peak_distances, 'o-', color='navy', markersize=6)
    plt.axhline(mean_distance, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_distance:.2f}')
    plt.xlabel('Test Number')
    plt.ylabel('Peak Distance (pixels)')
    plt.title('Peak Distance per Test')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Statistics summary
    plt.subplot(2, 2, 3)
    stats_labels = ['Mean', 'Std', 'Min', 'Max']
    stats_values = [mean_distance, std_distance, min_distance, max_distance]
    bars = plt.bar(stats_labels, stats_values, color=['red', 'orange', 'green', 'purple'])
    plt.ylabel('Distance (pixels)')
    plt.title('Peak Distance Statistics')
    for bar, value in zip(bars, stats_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom')

    # Subplot 4: Accuracy classification
    plt.subplot(2, 2, 4)
    excellent = sum(1 for d in peak_distances if d <= 1.0)
    good = sum(1 for d in peak_distances if 1.0 < d <= 2.0)
    fair = sum(1 for d in peak_distances if 2.0 < d <= 5.0)
    poor = sum(1 for d in peak_distances if d > 5.0)

    categories = ['Excellent\n(≤1px)', 'Good\n(1-2px)', 'Fair\n(2-5px)', 'Poor\n(>5px)']
    counts = [excellent, good, fair, poor]
    colors = ['green', 'yellow', 'orange', 'red']

    plt.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Peak Accuracy Classification')

    plt.tight_layout()
    plt.savefig(f'peak_accuracy_summary_sigma{sigma}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nSummary visualization saved to: peak_accuracy_summary_sigma{sigma}.png")

    return {
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'all_distances': peak_distances,
        'test_results': test_results
    }


if __name__ == "__main__":
    # Run the peak accuracy tests
    results = run_peak_accuracy_tests(num_tests=20, sigma=5)

    print("\nTest completed successfully!")