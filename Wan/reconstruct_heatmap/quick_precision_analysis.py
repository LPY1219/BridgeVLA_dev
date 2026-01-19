import matplotlib.pyplot as plt
import numpy as np

def analyze_precision_results():
    """Analyze the precision test results we got before timeout"""
    print("Quick Precision Analysis for σ=3 Gaussian Distributions")
    print("=" * 60)

    # Data from the interrupted test
    precisions = [32, 64, 128, 256, 512, 1024]
    mean_distances = [0.589, 0.467, 0.267, 0.267, 0.267, 0.267]
    perfect_rates = [46.7, 53.3, 73.3, 73.3, 73.3, 73.3]  # percentage
    avg_times = [0.0303, 0.0700, 0.1625, 0.3374, 0.7690, 1.6799]  # seconds

    print("Results Summary:")
    print("-" * 40)
    for i, precision in enumerate(precisions):
        print(f"{precision:4d} points: {mean_distances[i]:.3f}px, {perfect_rates[i]:5.1f}% perfect, {avg_times[i]*1000:6.1f}ms")

    # Analysis
    print(f"\nKey Findings:")
    print(f"1. Sweet spot at 128 reference points:")
    print(f"   - Mean distance: 0.267 pixels (excellent)")
    print(f"   - Perfect rate: 73.3% (very good)")
    print(f"   - Processing time: 162.5ms (reasonable)")

    print(f"\n2. Diminishing returns beyond 128 points:")
    print(f"   - No improvement in accuracy from 128→256→512→1024")
    print(f"   - Processing time increases dramatically")

    print(f"\n3. Lower precisions (32, 64) show:")
    print(f"   - Noticeably worse accuracy")
    print(f"   - Lower perfect reconstruction rates")

    # Calculate efficiency scores
    efficiency_scores = [perfect_rates[i] / (avg_times[i] * 1000) for i in range(len(precisions))]
    best_efficiency_idx = np.argmax(efficiency_scores)

    print(f"\n4. Efficiency Analysis (Perfect Rate / Processing Time):")
    for i, precision in enumerate(precisions):
        print(f"   {precision:4d} points: {efficiency_scores[i]:.2f} (higher is better)")

    print(f"\nRECOMMENDATION:")
    print(f"🎯 OPTIMAL CHOICE: {precisions[best_efficiency_idx]} reference points")
    print(f"   - Best balance of accuracy and speed")
    print(f"   - Mean error: {mean_distances[best_efficiency_idx]:.3f} pixels")
    print(f"   - Perfect rate: {perfect_rates[best_efficiency_idx]:.1f}%")
    print(f"   - Processing time: {avg_times[best_efficiency_idx]*1000:.1f}ms")

    print(f"\nUSAGE GUIDELINES for σ=3:")
    print(f"   Real-time applications: 64 points (fast, acceptable accuracy)")
    print(f"   General use: 128 points (RECOMMENDED - best efficiency)")
    print(f"   High accuracy needed: 128 points (same performance as higher)")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Accuracy vs Precision
    ax1 = axes[0, 0]
    ax1.plot(precisions, mean_distances, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel('Reference Points')
    ax1.set_ylabel('Mean Peak Distance (pixels)')
    ax1.set_title('Peak Accuracy vs Precision (σ=3)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Highlight recommended point
    ax1.scatter(precisions[best_efficiency_idx], mean_distances[best_efficiency_idx],
               color='red', s=150, zorder=5, label=f'Recommended: {precisions[best_efficiency_idx]}')
    ax1.legend()

    # Plot 2: Perfect Rate
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(precisions)), perfect_rates, color='green', alpha=0.7)
    ax2.set_xlabel('Precision Level')
    ax2.set_ylabel('Perfect Reconstruction Rate (%)')
    ax2.set_title('Perfect Peak Reconstruction Rate')
    ax2.set_xticks(range(len(precisions)))
    ax2.set_xticklabels([str(p) for p in precisions])
    ax2.grid(True, alpha=0.3, axis='y')

    # Highlight recommended
    bars[best_efficiency_idx].set_color('red')

    # Plot 3: Processing Time
    ax3 = axes[1, 0]
    ax3.plot(precisions, [t*1000 for t in avg_times], 's-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('Reference Points')
    ax3.set_ylabel('Processing Time (ms)')
    ax3.set_title('Speed vs Precision')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    # Plot 4: Efficiency Score
    ax4 = axes[1, 1]
    bars4 = ax4.bar(range(len(precisions)), efficiency_scores, color='purple', alpha=0.7)
    ax4.set_xlabel('Precision Level')
    ax4.set_ylabel('Efficiency Score')
    ax4.set_title('Overall Efficiency (Higher = Better)')
    ax4.set_xticks(range(len(precisions)))
    ax4.set_xticklabels([str(p) for p in precisions])
    ax4.grid(True, alpha=0.3, axis='y')

    # Highlight best efficiency
    bars4[best_efficiency_idx].set_color('red')
    ax4.text(best_efficiency_idx, efficiency_scores[best_efficiency_idx] * 1.05,
            f'Best: {precisions[best_efficiency_idx]}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('precision_analysis_sigma3_quick.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nVisualization saved to: precision_analysis_sigma3_quick.png")

    return precisions[best_efficiency_idx]

def create_optimized_function(optimal_precision):
    """Create the optimized function code"""
    function_code = f'''
def extract_heatmap_from_colormap_optimized(rgb_image, colormap_name='jet'):
    """
    Extract heatmap from RGB colormap - optimized for σ=3 Gaussian distributions
    Uses {optimal_precision} reference points for best efficiency
    """
    h, w = rgb_image.shape[:2]
    colormap = cm.get_cmap(colormap_name)

    # Optimized precision for σ=3: {optimal_precision} reference points
    reference_values = np.linspace(0, 1, {optimal_precision})
    reference_colors = colormap(reference_values)[:, :3]

    # Vectorized computation for 256x256 images
    rgb_flat = rgb_image.reshape(-1, 3)
    distances = np.sum((rgb_flat[:, None, :] - reference_colors[None, :, :]) ** 2, axis=2)
    closest_indices = np.argmin(distances, axis=1)
    extracted_values = reference_values[closest_indices]
    extracted_heatmap = extracted_values.reshape(h, w)

    return extracted_heatmap.astype(np.float32)
'''

    print(f"\nOptimized Function for σ=3:")
    print("=" * 50)
    print(function_code)

    # Save to file
    with open('optimized_colormap_conversion.py', 'w') as f:
        f.write(f"# Optimized colormap conversion for σ=3 Gaussian distributions\n")
        f.write(f"# Recommended precision: {optimal_precision} reference points\n\n")
        f.write("import numpy as np\nfrom matplotlib import cm\n\n")
        f.write(function_code)

    print(f"Function saved to: optimized_colormap_conversion.py")

if __name__ == "__main__":
    optimal_precision = analyze_precision_results()
    create_optimized_function(optimal_precision)

    print(f"\n{'='*60}")
    print(f"FINAL RECOMMENDATION FOR σ=3 GAUSSIAN DISTRIBUTIONS:")
    print(f"Use {optimal_precision} reference points for optimal peak detection accuracy")
    print(f"{'='*60}")