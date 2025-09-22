"""
可视化工具函数
包含图表生成、动画制作等辅助函数
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import cv2
from PIL import Image
import io
import base64
from typing import List, Tuple, Optional, Union, Dict
import os


def create_heatmap_overlay(rgb_image: np.ndarray,
                          heatmap: np.ndarray,
                          alpha: float = 0.6,
                          colormap: str = 'viridis') -> np.ndarray:
    """
    将heatmap叠加到RGB图像上

    Args:
        rgb_image: RGB图像 (H, W, 3) 范围[0, 1]
        heatmap: heatmap (H, W) 范围[0, 1]
        alpha: heatmap透明度
        colormap: 颜色映射名称

    Returns:
        叠加后的图像 (H, W, 3)
    """
    # 确保输入在正确范围内
    rgb_image = np.clip(rgb_image, 0, 1)
    heatmap = np.clip(heatmap, 0, 1)

    # 调整图像尺寸匹配
    if rgb_image.shape[:2] != heatmap.shape:
        heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]))

    # 将heatmap转换为彩色
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # 移除alpha通道

    # 叠加图像
    overlay = (1 - alpha) * rgb_image + alpha * heatmap_colored

    return np.clip(overlay, 0, 1)


def plot_heatmap_sequence(heatmap_sequence: Union[np.ndarray, torch.Tensor],
                         titles: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (15, 3),
                         colormap: str = 'viridis',
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制heatmap序列

    Args:
        heatmap_sequence: heatmap序列 (T, H, W)
        titles: 每个子图的标题
        figsize: 图像尺寸
        colormap: 颜色映射
        save_path: 保存路径

    Returns:
        matplotlib Figure对象
    """
    if torch.is_tensor(heatmap_sequence):
        heatmap_sequence = heatmap_sequence.cpu().numpy()

    T, H, W = heatmap_sequence.shape

    fig, axes = plt.subplots(1, T, figsize=figsize)
    if T == 1:
        axes = [axes]

    for t in range(T):
        im = axes[t].imshow(heatmap_sequence[t], cmap=colormap, vmin=0, vmax=1)
        axes[t].axis('off')

        if titles and t < len(titles):
            axes[t].set_title(titles[t])
        else:
            axes[t].set_title(f'Frame {t+1}')

    # 添加颜色条
    fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_comparison_grid(pred_sequence: Union[np.ndarray, torch.Tensor],
                        gt_sequence: Union[np.ndarray, torch.Tensor],
                        rgb_image: Optional[np.ndarray] = None,
                        figsize: Tuple[int, int] = (15, 6),
                        colormap: str = 'viridis',
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制预测与ground truth的对比网格

    Args:
        pred_sequence: 预测序列 (T, H, W)
        gt_sequence: ground truth序列 (T, H, W)
        rgb_image: 可选的RGB图像 (H, W, 3)
        figsize: 图像尺寸
        colormap: 颜色映射
        save_path: 保存路径

    Returns:
        matplotlib Figure对象
    """
    if torch.is_tensor(pred_sequence):
        pred_sequence = pred_sequence.cpu().numpy()
    if torch.is_tensor(gt_sequence):
        gt_sequence = gt_sequence.cpu().numpy()

    T = pred_sequence.shape[0]
    rows = 3 if rgb_image is not None else 2
    cols = T

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for t in range(T):
        # 预测heatmap
        im1 = axes[0, t].imshow(pred_sequence[t], cmap=colormap, vmin=0, vmax=1)
        axes[0, t].set_title(f'Pred Frame {t+1}')
        axes[0, t].axis('off')

        # Ground truth heatmap
        im2 = axes[1, t].imshow(gt_sequence[t], cmap=colormap, vmin=0, vmax=1)
        axes[1, t].set_title(f'GT Frame {t+1}')
        axes[1, t].axis('off')

        # RGB叠加 (如果提供)
        if rgb_image is not None:
            overlay = create_heatmap_overlay(rgb_image, pred_sequence[t])
            axes[2, t].imshow(overlay)
            axes[2, t].set_title(f'Overlay Frame {t+1}')
            axes[2, t].axis('off')

    # 添加颜色条
    fig.colorbar(im1, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_heatmap_animation(heatmap_sequence: Union[np.ndarray, torch.Tensor],
                           rgb_image: Optional[np.ndarray] = None,
                           fps: int = 2,
                           colormap: str = 'viridis',
                           save_path: Optional[str] = None) -> animation.FuncAnimation:
    """
    创建heatmap序列动画

    Args:
        heatmap_sequence: heatmap序列 (T, H, W)
        rgb_image: 可选的背景RGB图像
        fps: 帧率
        colormap: 颜色映射
        save_path: 保存路径 (.gif 或 .mp4)

    Returns:
        matplotlib动画对象
    """
    if torch.is_tensor(heatmap_sequence):
        heatmap_sequence = heatmap_sequence.cpu().numpy()

    T, H, W = heatmap_sequence.shape

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    # 初始化图像
    if rgb_image is not None:
        # 使用叠加模式
        initial_overlay = create_heatmap_overlay(rgb_image, heatmap_sequence[0])
        im = ax.imshow(initial_overlay)
    else:
        # 只显示heatmap
        im = ax.imshow(heatmap_sequence[0], cmap=colormap, vmin=0, vmax=1)

    # 添加标题
    title = ax.text(0.5, 1.02, 'Frame 1', transform=ax.transAxes,
                   ha='center', va='bottom', fontsize=14)

    def animate(frame):
        if rgb_image is not None:
            overlay = create_heatmap_overlay(rgb_image, heatmap_sequence[frame])
            im.set_array(overlay)
        else:
            im.set_array(heatmap_sequence[frame])

        title.set_text(f'Frame {frame + 1}')
        return [im, title]

    # 创建动画
    anim = animation.FuncAnimation(
        fig, animate, frames=T, interval=1000//fps, blit=True, repeat=True
    )

    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)

    return anim


def plot_peak_trajectory(heatmap_sequence: Union[np.ndarray, torch.Tensor],
                        rgb_image: Optional[np.ndarray] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制heatmap序列中峰值点的运动轨迹

    Args:
        heatmap_sequence: heatmap序列 (T, H, W)
        rgb_image: 可选的背景RGB图像
        figsize: 图像尺寸
        save_path: 保存路径

    Returns:
        matplotlib Figure对象
    """
    try:
        from .heatmap_utils import find_global_peak
    except ImportError:
        from heatmap_utils import find_global_peak

    if torch.is_tensor(heatmap_sequence):
        heatmap_sequence = heatmap_sequence.cpu().numpy()

    T, H, W = heatmap_sequence.shape

    # 提取峰值轨迹
    trajectory = []
    for t in range(T):
        peak = find_global_peak(heatmap_sequence[t])
        trajectory.append(peak)

    trajectory = np.array(trajectory)  # Shape: (T, 2) - (y, x)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 显示背景
    if rgb_image is not None:
        ax.imshow(rgb_image, extent=[0, W, H, 0])
    else:
        # 显示最后一帧heatmap作为背景
        ax.imshow(heatmap_sequence[-1], cmap='viridis', alpha=0.3, extent=[0, W, H, 0])

    # 绘制轨迹
    ax.plot(trajectory[:, 1], trajectory[:, 0], 'ro-', linewidth=2, markersize=8,
           label='Peak Trajectory', alpha=0.8)

    # 标记起点和终点
    ax.plot(trajectory[0, 1], trajectory[0, 0], 'go', markersize=12, label='Start')
    ax.plot(trajectory[-1, 1], trajectory[-1, 0], 'bo', markersize=12, label='End')

    # 添加帧编号
    for t, (y, x) in enumerate(trajectory):
        ax.annotate(f'{t+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, color='white', weight='bold')

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # 翻转y轴
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Heatmap Peak Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_error_analysis(pred_sequence: Union[np.ndarray, torch.Tensor],
                       gt_sequence: Union[np.ndarray, torch.Tensor],
                       figsize: Tuple[int, int] = (15, 5),
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    绘制误差分析图表

    Args:
        pred_sequence: 预测序列 (T, H, W)
        gt_sequence: ground truth序列 (T, H, W)
        figsize: 图像尺寸
        save_path: 保存路径

    Returns:
        matplotlib Figure对象
    """
    try:
        from .heatmap_utils import find_global_peak, calculate_mse_loss
    except ImportError:
        from heatmap_utils import find_global_peak, calculate_mse_loss
    from scipy.spatial.distance import euclidean

    if torch.is_tensor(pred_sequence):
        pred_sequence = pred_sequence.cpu().numpy()
    if torch.is_tensor(gt_sequence):
        gt_sequence = gt_sequence.cpu().numpy()

    T = pred_sequence.shape[0]

    # 计算各种误差指标
    mse_errors = []
    peak_distances = []

    for t in range(T):
        # MSE误差
        mse = calculate_mse_loss(pred_sequence[t], gt_sequence[t])
        mse_errors.append(mse)

        # 峰值距离误差
        pred_peak = find_global_peak(pred_sequence[t])
        gt_peak = find_global_peak(gt_sequence[t])
        distance = euclidean(pred_peak, gt_peak)
        peak_distances.append(distance)

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. MSE随时间变化
    axes[0].plot(range(1, T+1), mse_errors, 'bo-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE Error over Time')
    axes[0].grid(True, alpha=0.3)

    # 2. 峰值距离随时间变化
    axes[1].plot(range(1, T+1), peak_distances, 'ro-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Peak Distance (pixels)')
    axes[1].set_title('Peak Distance Error over Time')
    axes[1].grid(True, alpha=0.3)

    # 3. 误差分布直方图
    axes[2].hist(peak_distances, bins=min(10, T), alpha=0.7, color='green', edgecolor='black')
    axes[2].set_xlabel('Peak Distance (pixels)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Peak Distance Distribution')
    axes[2].grid(True, alpha=0.3)

    # 添加统计信息
    mean_mse = np.mean(mse_errors)
    mean_distance = np.mean(peak_distances)
    axes[0].axhline(y=mean_mse, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_mse:.4f}')
    axes[1].axhline(y=mean_distance, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_distance:.2f}')

    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_visualization_report(results: Dict,
                            output_dir: str,
                            experiment_name: str = "heatmap_prediction") -> str:
    """
    保存完整的可视化报告

    Args:
        results: 包含预测结果的字典
        output_dir: 输出目录
        experiment_name: 实验名称

    Returns:
        报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 提取数据
    pred_sequence = results['pred_sequence']
    gt_sequence = results['gt_sequence']
    rgb_image = results.get('rgb_image', None)
    instruction = results.get('instruction', "No instruction")

    # 创建各种可视化
    print("Generating visualizations...")

    # 1. 对比网格
    fig1 = plot_comparison_grid(pred_sequence, gt_sequence, rgb_image)
    comparison_path = os.path.join(output_dir, f"{experiment_name}_comparison.png")
    fig1.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # 2. 动画
    anim_path = os.path.join(output_dir, f"{experiment_name}_animation.gif")
    anim = create_heatmap_animation(pred_sequence, rgb_image, save_path=anim_path)
    plt.close()

    # 3. 轨迹图
    fig3 = plot_peak_trajectory(pred_sequence, rgb_image)
    trajectory_path = os.path.join(output_dir, f"{experiment_name}_trajectory.png")
    fig3.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # 4. 误差分析
    fig4 = plot_error_analysis(pred_sequence, gt_sequence)
    error_path = os.path.join(output_dir, f"{experiment_name}_error_analysis.png")
    fig4.savefig(error_path, dpi=150, bbox_inches='tight')
    plt.close(fig4)

    # 5. 生成HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heatmap Prediction Report - {experiment_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 40px; }}
            .image {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .instruction {{ background-color: #f5f5f5; padding: 15px; border-left: 4px solid #007cba; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Heatmap Prediction Report</h1>
            <h2>{experiment_name}</h2>
        </div>

        <div class="section">
            <h3>Task Instruction</h3>
            <div class="instruction">
                {instruction}
            </div>
        </div>

        <div class="section">
            <h3>Prediction vs Ground Truth Comparison</h3>
            <div class="image">
                <img src="{os.path.basename(comparison_path)}" alt="Comparison Grid">
            </div>
        </div>

        <div class="section">
            <h3>Peak Trajectory</h3>
            <div class="image">
                <img src="{os.path.basename(trajectory_path)}" alt="Peak Trajectory">
            </div>
        </div>

        <div class="section">
            <h3>Error Analysis</h3>
            <div class="image">
                <img src="{os.path.basename(error_path)}" alt="Error Analysis">
            </div>
        </div>

        <div class="section">
            <h3>Animation</h3>
            <div class="image">
                <img src="{os.path.basename(anim_path)}" alt="Heatmap Animation">
            </div>
        </div>
    </body>
    </html>
    """

    # 保存HTML报告
    report_path = os.path.join(output_dir, f"{experiment_name}_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"Visualization report saved to: {report_path}")
    return report_path


def fig_to_base64(fig: plt.Figure) -> str:
    """
    将matplotlib图形转换为base64字符串

    Args:
        fig: matplotlib Figure对象

    Returns:
        base64编码的图像字符串
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return img_base64


def test_visualization_utils():
    """
    测试可视化工具函数
    """
    print("Testing visualization utilities...")

    # 创建测试数据
    T, H, W = 5, 64, 64

    # 创建移动峰值的heatmap序列
    heatmap_sequence = np.zeros((T, H, W))
    for t in range(T):
        center_x = int(20 + t * 5)
        center_y = int(32 + t * 2)
        y, x = np.ogrid[:H, :W]
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 5**2))
        heatmap_sequence[t] = heatmap

    # 创建RGB图像
    rgb_image = np.random.rand(H, W, 3) * 0.3 + 0.2

    # 测试可视化函数
    print("Creating comparison plot...")
    fig1 = plot_comparison_grid(heatmap_sequence, heatmap_sequence, rgb_image)
    plt.close(fig1)

    print("Creating trajectory plot...")
    fig2 = plot_peak_trajectory(heatmap_sequence, rgb_image)
    plt.close(fig2)

    print("Creating error analysis...")
    # 添加一些噪声作为"预测"
    pred_sequence = heatmap_sequence + np.random.normal(0, 0.1, heatmap_sequence.shape)
    fig3 = plot_error_analysis(pred_sequence, heatmap_sequence)
    plt.close(fig3)

    print("Visualization utils test completed!")
    return True


if __name__ == "__main__":
    # 运行测试
    test_visualization_utils()