#!/usr/bin/env python3
"""
检查所有连接的ZED相机
"""
import pyzed.sl as sl

def check_zed_cameras():
    """检查所有连接的ZED相机"""
    print("🔍 正在扫描ZED相机...\n")

    # 获取所有相机信息
    cameras = sl.Camera.get_device_list()

    if len(cameras) == 0:
        print("❌ 未检测到任何ZED相机！")
        print("\n请检查：")
        print("  1. ZED相机是否已连接到USB端口")
        print("  2. USB线缆是否正常工作")
        print("  3. 是否使用USB 3.0端口（蓝色接口）")
        print("  4. ZED相机指示灯是否亮起")
        print("  5. 运行 'lsusb' 命令查看USB设备列表")
        return []

    print(f"✅ 检测到 {len(cameras)} 个ZED相机\n")
    print("=" * 70)

    serial_numbers = []
    for i, cam_info in enumerate(cameras):
        serial = cam_info.serial_number
        model = cam_info.camera_model
        state = cam_info.camera_state

        serial_numbers.append(serial)

        print(f"📷 ZED相机 #{i+1}")
        print(f"   型号: {model}")
        print(f"   序列号: {serial}")
        print(f"   状态: {state}")
        print("-" * 70)

    print("\n📋 所有序列号列表:")
    for i, sn in enumerate(serial_numbers):
        print(f"   {i+1}. {sn}")

    print("\n💡 使用提示:")
    print("   在 real_camera_utils_lpy.py 第339-341行修改序列号：")
    for i, sn in enumerate(serial_numbers[:3]):
        cam_name = ["top上相机", "右边相机", "top下相机"][i]
        print(f"   static_serial_number_{i+1} = {sn}  # {cam_name}")

    return serial_numbers

if __name__ == "__main__":
    serials = check_zed_cameras()
