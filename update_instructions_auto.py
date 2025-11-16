#!/usr/bin/env python3
"""
批量更新instruction.txt文件内容（自动版本，无需确认）
"""
import os
from pathlib import Path

def update_instruction_files(base_dir, new_instruction):
    """
    更新指定目录下所有trial的instruction.txt文件

    Args:
        base_dir: 基础目录路径
        new_instruction: 新的指令内容
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"错误：目录不存在: {base_dir}")
        return False

    # 统计
    total_files = 0
    updated_files = 0
    failed_files = 0
    skipped_files = 0

    print(f"开始扫描目录: {base_dir}\n")

    # 遍历所有子目录
    for trial_dir in sorted(base_path.iterdir()):
        if not trial_dir.is_dir():
            continue

        # 检查是否是trial目录（通常以trail_开头）
        if not trial_dir.name.startswith('trail_'):
            continue

        instruction_file = trial_dir / 'instruction.txt'

        if instruction_file.exists():
            total_files += 1
            try:
                # 读取原内容
                with open(instruction_file, 'r', encoding='utf-8') as f:
                    old_content = f.read().strip()

                # 如果内容已经是目标内容，跳过
                if old_content == new_instruction:
                    skipped_files += 1
                    print(f"⊙ {trial_dir.name}/instruction.txt - 内容已是最新，跳过")
                    continue

                # 写入新内容
                with open(instruction_file, 'w', encoding='utf-8') as f:
                    f.write(new_instruction)

                updated_files += 1
                print(f"✓ {trial_dir.name}/instruction.txt - 已更新")
                if old_content:
                    print(f"  旧: '{old_content}' -> 新: '{new_instruction}'")

            except Exception as e:
                failed_files += 1
                print(f"✗ {trial_dir.name}/instruction.txt - 错误: {e}")
        else:
            print(f"⚠ {trial_dir.name} - instruction.txt不存在")

    # 打印统计信息
    print("\n" + "="*70)
    print("更新完成！")
    print("="*70)
    print(f"总共找到:   {total_files} 个instruction.txt文件")
    print(f"成功更新:   {updated_files} 个")
    print(f"已是最新:   {skipped_files} 个")
    print(f"更新失败:   {failed_files} 个")
    print("="*70)

    return failed_files == 0


if __name__ == "__main__":
    # 配置
    BASE_DIR = "/data/Franka_data/put_the_lion_on_the_top_shelf"
    NEW_INSTRUCTION = "put the lion on the top shelf"

    print("="*70)
    print("批量更新instruction.txt文件（自动模式）")
    print("="*70)
    print(f"目标目录:   {BASE_DIR}")
    print(f"新指令内容: {NEW_INSTRUCTION}")
    print("="*70)
    print()

    success = update_instruction_files(BASE_DIR, NEW_INSTRUCTION)

    if success:
        print("\n✓ 所有文件更新成功！")
        exit(0)
    else:
        print("\n✗ 部分文件更新失败，请检查错误信息")
        exit(1)
