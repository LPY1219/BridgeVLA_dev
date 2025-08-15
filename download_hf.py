#!/usr/bin/env python3
"""
Hugging Face Dataset/Model File Downloader with Mirror Support and Auto-Retry

This script downloads specific files from Hugging Face datasets or models
using mirror sites for improved accessibility and speed, with automatic retry
on network failures.

Usage:
    # 脚本会自动使用镜像站 hf-mirror.com（适合中国大陆用户）
    python download_hf.py --repo_id "LPY/BridgeVLA_RLBench_TRAIN_BUFFER" \
                          --filename "place_shape_in_shape_sorter.tar.xz" \
                          --save_dir "/share/project/lpy/BridgeVLA/data/RLBench/replay_buffer" \
                          --repo_type "dataset"

Author: LPY
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from urllib.parse import urljoin
from tqdm import tqdm
import time
import signal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 自动设置镜像站环境变量（适合中国大陆用户）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Hugging Face mirror sites (commonly used in China)
HF_MIRRORS = [
    "https://hf-mirror.com",  # Popular mirror in China
    "https://huggingface.co", # Original site (fallback)
]

class HuggingFaceDownloader:
    def __init__(self, use_mirror=True, verbose=True, max_retries=5, retry_delay=1):
        self.use_mirror = use_mirror
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = self._create_session()
        
        # Check for HF_ENDPOINT environment variable
        self.hf_endpoint = os.environ.get('HF_ENDPOINT')
        if self.hf_endpoint and self.verbose:
            print(f"Using HF_ENDPOINT from environment: {self.hf_endpoint}")
            
    def _create_session(self):
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        # Define retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=2
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeouts
        session.timeout = (30, 300)  # (connect_timeout, read_timeout)
        
        return session
        
    def _get_download_url(self, repo_id, filename, repo_type="dataset", revision="main"):
        """
        Construct download URL for HF files
        
        Args:
            repo_id: Repository ID (e.g., "LPY/BridgeVLA_RLBench_TRAIN_BUFFER")
            filename: File name to download (e.g., "place_shape_in_shape_sorter.tar.xz")
            repo_type: "dataset" or "model"
            revision: Branch/tag (default: "main")
        """
        if repo_type == "dataset":
            path = f"datasets/{repo_id}/resolve/{revision}/{filename}"
        elif repo_type == "model":
            path = f"{repo_id}/resolve/{revision}/{filename}"
        else:
            raise ValueError(f"Unsupported repo_type: {repo_type}")
            
        return path
    
    def _try_download_from_mirrors(self, url_path, save_path):
        """
        Try downloading from different mirror sites with auto-retry
        """
        # Use HF_ENDPOINT if set, otherwise use mirrors
        if self.hf_endpoint:
            mirrors = [self.hf_endpoint]
        else:
            mirrors = HF_MIRRORS if self.use_mirror else [HF_MIRRORS[-1]]
        
        last_exception = None
        
        for mirror_idx, mirror_base in enumerate(mirrors):
            full_url = urljoin(mirror_base, url_path)
            
            if self.verbose:
                print(f"[{mirror_idx + 1}/{len(mirrors)}] Trying to download from: {mirror_base}")
                print(f"Full URL: {full_url}")
            
            for attempt in range(self.max_retries):
                try:
                    return self._download_file_with_resume(full_url, save_path)
                except KeyboardInterrupt:
                    print("\nDownload interrupted by user")
                    raise
                except Exception as e:
                    last_exception = e
                    if self.verbose:
                        print(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                    
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        if self.verbose:
                            print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    continue
            
            if self.verbose:
                print(f"Failed to download from {mirror_base} after {self.max_retries} attempts")
        
        raise Exception(f"Failed to download from all mirror sites. Last error: {last_exception}")
    
    def _download_file_with_resume(self, url, save_path):
        """
        Download file with progress bar and resume capability
        """
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Check if partial file exists
        temp_path = save_path + '.download'
        resume_pos = 0
        
        if os.path.exists(temp_path):
            resume_pos = os.path.getsize(temp_path)
            if self.verbose:
                print(f"Resuming download from byte {resume_pos}")
        
        # Prepare headers for resume
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
        
        # Make request with resume headers
        response = self.session.get(url, headers=headers, stream=True, timeout=(30, 300))
        response.raise_for_status()
        
        # Get file size information
        if resume_pos > 0 and response.status_code == 206:
            # Partial content response
            content_range = response.headers.get('content-range', '')
            if content_range:
                total_size = int(content_range.split('/')[-1])
            else:
                total_size = resume_pos + int(response.headers.get('content-length', 0))
        else:
            # Full content response
            total_size = int(response.headers.get('content-length', 0))
            resume_pos = 0  # Start from beginning if server doesn't support resume
        
        # Open file in appropriate mode
        mode = 'ab' if resume_pos > 0 and response.status_code == 206 else 'wb'
        
        # Download with progress bar
        with open(temp_path, mode) as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            initial=resume_pos,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
                # Download completed, move temp file to final location
                if os.path.exists(save_path):
                    os.remove(save_path)
                os.rename(temp_path, save_path)
                
            except Exception as e:
                # Keep temp file for resume
                if self.verbose:
                    print(f"Download interrupted, temp file kept at: {temp_path}")
                raise e
        
        if self.verbose:
            print(f"Successfully downloaded: {save_path}")
        
        return save_path
    
    def download(self, repo_id, filename, save_dir, repo_type="dataset", revision="main"):
        """
        Download a specific file from Hugging Face
        
        Args:
            repo_id: Repository ID (e.g., "LPY/BridgeVLA_RLBench_TRAIN_BUFFER")
            filename: File name to download
            save_dir: Directory to save the file
            repo_type: "dataset" or "model"
            revision: Branch/tag
        """
        # Construct download URL path
        url_path = self._get_download_url(repo_id, filename, repo_type, revision)
        
        # Create full save path
        save_path = os.path.join(save_dir, filename)
        
        if self.verbose:
            print(f"Downloading {filename} from {repo_id}")
            print(f"Save to: {save_path}")
        
        # Check if file already exists
        if os.path.exists(save_path):
            print(f"File already exists: {save_path}")
            overwrite = input("Overwrite? (y/N): ").lower().strip()
            if overwrite != 'y':
                print("Download cancelled.")
                return save_path
        
        # Download file
        return self._try_download_from_mirrors(url_path, save_path)
    
    def download_multiple(self, downloads):
        """
        Download multiple files
        
        Args:
            downloads: List of download configs, each containing:
                      {
                          "repo_id": str,
                          "filename": str, 
                          "save_dir": str,
                          "repo_type": str (optional, default "dataset"),
                          "revision": str (optional, default "main")
                      }
        """
        results = []
        for i, config in enumerate(downloads, 1):
            print(f"\n[{i}/{len(downloads)}] Processing download...")
            try:
                result = self.download(**config)
                results.append({"status": "success", "path": result, "config": config})
            except Exception as e:
                print(f"Failed to download {config.get('filename', 'unknown')}: {e}")
                results.append({"status": "failed", "error": str(e), "config": config})
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Download files from Hugging Face with mirror support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 下载数据集文件 (自动使用镜像站)
  python download_hf.py --repo_id "LPY/BridgeVLA_RLBench_TRAIN_BUFFER" \\
                        --filename "place_shape_in_shape_sorter.tar.xz" \\
                        --save_dir "/share/project/lpy/BridgeVLA/data/RLBench/replay_buffer"
  
  # 下载模型文件
  python download_hf.py --repo_id "microsoft/DialoGPT-medium" \\
                        --filename "pytorch_model.bin" \\
                        --save_dir "./models" \\
                        --repo_type "model"
  
  # 增加重试次数以应对网络不稳定
  python download_hf.py --repo_id "LPY/BridgeVLA_RLBench_TRAIN_BUFFER" \\
                        --filename "place_shape_in_shape_sorter.tar.xz" \\
                        --save_dir "/share/project/lpy/BridgeVLA/data/RLBench/replay_buffer" \\
                        --max-retries 10 --retry-delay 2
  
  # 禁用镜像站，只使用原站
  python download_hf.py --repo_id "LPY/BridgeVLA_RLBench_TRAIN_BUFFER" \\
                        --filename "place_shape_in_shape_sorter.tar.xz" \\
                        --save_dir "/share/project/lpy/BridgeVLA/data/RLBench/replay_buffer" \\
                        --no-mirror
        """
    )
    
    parser.add_argument("--repo_id", required=True,
                       help="Repository ID (e.g., 'LPY/BridgeVLA_RLBench_TRAIN_BUFFER')")
    parser.add_argument("--filename", required=True,
                       help="File name to download (e.g., 'place_shape_in_shape_sorter.tar.xz')")
    parser.add_argument("--save_dir", required=True,
                       help="Directory to save the downloaded file")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model"],
                       help="Repository type (default: dataset)")
    parser.add_argument("--revision", default="main",
                       help="Branch or tag to download from (default: main)")
    parser.add_argument("--no-mirror", action="store_true",
                       help="Disable mirror sites, use original HF only")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    parser.add_argument("--max-retries", type=int, default=5,
                       help="Maximum number of retry attempts (default: 5)")
    parser.add_argument("--retry-delay", type=int, default=1,
                       help="Initial retry delay in seconds (default: 1)")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = HuggingFaceDownloader(
        use_mirror=not args.no_mirror,
        verbose=not args.quiet,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )
    
    try:
        # Download file
        result_path = downloader.download(
            repo_id=args.repo_id,
            filename=args.filename,
            save_dir=args.save_dir,
            repo_type=args.repo_type,
            revision=args.revision
        )
        
        print(f"\n✅ Download completed successfully!")
        print(f"File saved to: {result_path}")
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


# Example usage as a module
def download_rlbench_data():
    """
    Example function to download RLBench training data
    """
    downloader = HuggingFaceDownloader()
    
    # Define downloads for RLBench data
    downloads = [
        {
            "repo_id": "LPY/BridgeVLA_RLBench_TRAIN_BUFFER",
            "filename": "place_shape_in_shape_sorter.tar.xz",
            "save_dir": "/share/project/lpy/BridgeVLA/data/RLBench/replay_buffer"
        },
        # Add more files as needed
        # {
        #     "repo_id": "LPY/BridgeVLA_RLBench_TRAIN_BUFFER", 
        #     "filename": "close_jar.tar.xz",
        #     "save_dir": "/share/project/lpy/BridgeVLA/data/RLBench/replay_buffer"
        # }
    ]
    
    return downloader.download_multiple(downloads)


if __name__ == "__main__":
    main()
