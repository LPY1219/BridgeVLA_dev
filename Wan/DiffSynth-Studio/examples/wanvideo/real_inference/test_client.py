"""
Test client for RoboWan Server
Demonstrates how to send requests and receive predictions
"""

import requests
from PIL import Image
import io
import numpy as np
from typing import List


class RoboWanClient:
    """Client for communicating with RoboWan server"""

    def __init__(self, server_url: str = "http://localhost:5555"):
        """
        Initialize client

        Args:
            server_url: URL of the server (e.g., "http://localhost:5555")
        """
        self.server_url = server_url
        self.predict_url = f"{server_url}/predict"
        self.health_url = f"{server_url}/health"

    def check_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def predict(
        self,
        heatmap_images: List[Image.Image],
        rgb_images: List[Image.Image],
        prompt: str,
        initial_rotation: List[float],
        initial_gripper: int,
        num_frames: int = 12
    ) -> dict:
        """
        Send prediction request to server

        Args:
            heatmap_images: List of PIL Images for heatmap (multi-view)
            rgb_images: List of PIL Images for RGB (multi-view)
            prompt: Task instruction
            initial_rotation: Initial rotation [roll, pitch, yaw] in degrees
            initial_gripper: Initial gripper state (0 or 1)
            num_frames: Number of frames to predict

        Returns:
            Dictionary containing:
                - success: bool
                - rotation: List[List[float]] - (num_frames, 3)
                - gripper: List[int] - (num_frames,)
                - error: str (if failed)
        """
        # Prepare files
        files = []

        # Add heatmap images
        for i, img in enumerate(heatmap_images):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            files.append(('heatmap_images', (f'heatmap_{i}.png', buffer, 'image/png')))

        # Add RGB images
        for i, img in enumerate(rgb_images):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            files.append(('rgb_images', (f'rgb_{i}.png', buffer, 'image/png')))

        # Prepare form data
        rotation_str = ','.join(map(str, initial_rotation))
        data = {
            'prompt': prompt,
            'initial_rotation': rotation_str,
            'initial_gripper': initial_gripper,
            'num_frames': num_frames
        }

        try:
            # Send request
            response = requests.post(self.predict_url, files=files, data=data, timeout=60)

            # Parse response
            result = response.json()
            return result

        except Exception as e:
            print(f"Request failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Test the client"""
    # Initialize client
    client = RoboWanClient(server_url="http://localhost:5555")

    # Check server health
    print("Checking server health...")
    if not client.check_health():
        print("❌ Server is not healthy!")
        return
    print("✓ Server is healthy")

    # Create dummy images (replace with real images in practice)
    print("\nCreating dummy test images...")
    num_views = 3
    heatmap_images = [Image.new('RGB', (256, 256), color=(255, 0, 0)) for _ in range(num_views)]
    rgb_images = [Image.new('RGB', (256, 256), color=(0, 255, 0)) for _ in range(num_views)]

    # Set initial conditions
    prompt = "put the lion on the top shelf"
    initial_rotation = [-180.0, 0.0, 0.0]  # [roll, pitch, yaw] in degrees
    initial_gripper = 1  # 0 or 1
    num_frames = 13 # 加上了初始帧

    print(f"\nSending prediction request...")
    print(f"  Prompt: {prompt}")
    print(f"  Initial rotation: {initial_rotation}")
    print(f"  Initial gripper: {initial_gripper}")
    print(f"  Num frames: {num_frames}")

    # Send request
    result = client.predict(
        heatmap_images=heatmap_images,
        rgb_images=rgb_images,
        prompt=prompt,
        initial_rotation=initial_rotation,
        initial_gripper=initial_gripper,
        num_frames=num_frames
    )

    # Print result
    print("\n" + "="*50)
    if result['success']:
        print("✓ Prediction successful!")

        # Print position predictions if available
        if 'position' in result and result['position'] is not None:
            print(f"\nPosition predictions ({len(result['position'])} frames):")
            for i, pos in enumerate(result['position']):
                print(f"  Frame {i}: X={pos[0]:.4f}m, Y={pos[1]:.4f}m, Z={pos[2]:.4f}m")
        else:
            print("\nPosition predictions: Not available")

        print(f"\nRotation predictions ({len(result['rotation'])} frames):")
        for i, rot in enumerate(result['rotation']):
            print(f"  Frame {i}: Roll={rot[0]:.1f}°, Pitch={rot[1]:.1f}°, Yaw={rot[2]:.1f}°")

        print(f"\nGripper predictions ({len(result['gripper'])} frames):")
        print(f"  {result['gripper']}")
    else:
        print("❌ Prediction failed!")
        print(f"Error: {result['error']}")
    print("="*50)


if __name__ == "__main__":
    main()
