"""
Update extrinsics.pkl files in dataset with the latest camera extrinsics.

This script reads the current camera extrinsics from real_camera_utils_lpy.py
and updates all extrinsics.pkl files in the specified dataset directory.

Usage:
    python update_extrinsics.py --dataset_path /path/to/dataset
    python update_extrinsics.py --dataset_path /path/to/dataset --dry-run  # Preview without changes
    python update_extrinsics.py --dataset_path /path/to/dataset --backup   # Create backups before updating
"""

import os
import sys
import pickle
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from real_camera_utils_lpy import get_cam_extrinsic


def get_new_extrinsics():
    """Get the latest camera extrinsics from real_camera_utils_lpy.py"""
    extrinsics = {
        '3rd_1': get_cam_extrinsic('3rd_1'),
        '3rd_2': get_cam_extrinsic('3rd_2'),
        '3rd_3': get_cam_extrinsic('3rd_3'),
    }
    return extrinsics


def find_all_trails(dataset_path):
    """Find all trail directories in the dataset"""
    dataset_path = Path(dataset_path)
    trails = []

    # Check if dataset_path itself contains trail directories
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and item.name.startswith('trail_'):
            trails.append(item)

    return trails


def update_extrinsics_file(trail_path, new_extrinsics, dry_run=False, backup=False):
    """
    Update the extrinsics.pkl file in a trail directory

    Args:
        trail_path: Path to trail directory
        new_extrinsics: New extrinsics dictionary
        dry_run: If True, only print what would be done without making changes
        backup: If True, create a backup of the original file

    Returns:
        True if successful, False otherwise
    """
    extrinsics_path = trail_path / 'extrinsics.pkl'

    if not extrinsics_path.exists():
        print(f"  [WARNING] extrinsics.pkl not found in {trail_path}")
        return False

    # Read old extrinsics for comparison
    with open(extrinsics_path, 'rb') as f:
        old_extrinsics = pickle.load(f)

    if dry_run:
        print(f"  [DRY-RUN] Would update: {extrinsics_path}")
        print(f"    Old 3rd_1:\n{old_extrinsics['3rd_1']}")
        print(f"    New 3rd_1:\n{new_extrinsics['3rd_1']}")
        return True

    # Create backup if requested
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = trail_path / f'extrinsics_backup_{timestamp}.pkl'
        shutil.copy(extrinsics_path, backup_path)
        print(f"  [BACKUP] Created: {backup_path}")

    # Write new extrinsics
    with open(extrinsics_path, 'wb') as f:
        pickle.dump(new_extrinsics, f)

    print(f"  [UPDATED] {extrinsics_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update extrinsics.pkl files in dataset with latest camera extrinsics"
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to dataset directory containing trail_* subdirectories'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview changes without actually updating files'
    )
    parser.add_argument(
        '--backup', action='store_true',
        help='Create backup of original extrinsics.pkl before updating'
    )
    parser.add_argument(
        '--show-diff', action='store_true',
        help='Show detailed difference between old and new extrinsics'
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return 1

    print("=" * 60)
    print("Extrinsics Update Tool")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Dry run: {args.dry_run}")
    print(f"Backup: {args.backup}")
    print()

    # Get new extrinsics
    print("Loading new extrinsics from real_camera_utils_lpy.py...")
    new_extrinsics = get_new_extrinsics()

    print("\nNew extrinsics:")
    for cam_type, matrix in new_extrinsics.items():
        print(f"  {cam_type}:")
        print(f"    Translation: [{matrix[0,3]:.6f}, {matrix[1,3]:.6f}, {matrix[2,3]:.6f}]")
    print()

    # Find all trails
    trails = find_all_trails(dataset_path)

    if not trails:
        print(f"No trail directories found in {dataset_path}")
        return 1

    print(f"Found {len(trails)} trail directories:")
    for trail in trails:
        print(f"  - {trail.name}")
    print()

    # Show diff if requested (only for first trail)
    if args.show_diff and trails:
        first_trail = trails[0]
        extrinsics_path = first_trail / 'extrinsics.pkl'
        if extrinsics_path.exists():
            with open(extrinsics_path, 'rb') as f:
                old_extrinsics = pickle.load(f)

            print("Detailed comparison (from first trail):")
            print("-" * 60)
            for cam_type in ['3rd_1', '3rd_2', '3rd_3']:
                print(f"\n{cam_type}:")
                print(f"  Old matrix:\n{old_extrinsics[cam_type]}")
                print(f"  New matrix:\n{new_extrinsics[cam_type]}")

                # Check if matrices are different
                import numpy as np
                if np.allclose(old_extrinsics[cam_type], new_extrinsics[cam_type]):
                    print(f"  Status: SAME (no change needed)")
                else:
                    print(f"  Status: DIFFERENT (will be updated)")
            print("-" * 60)
            print()

    # Update each trail
    print("Updating extrinsics files...")
    success_count = 0
    fail_count = 0

    for trail in trails:
        print(f"\nProcessing {trail.name}...")
        if update_extrinsics_file(trail, new_extrinsics, args.dry_run, args.backup):
            success_count += 1
        else:
            fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total trails: {len(trails)}")
    print(f"Successfully {'would update' if args.dry_run else 'updated'}: {success_count}")
    print(f"Failed: {fail_count}")

    if args.dry_run:
        print("\n[NOTE] This was a dry run. No files were actually modified.")
        print("       Remove --dry-run flag to apply changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
