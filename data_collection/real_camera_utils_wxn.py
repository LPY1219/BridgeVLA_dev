import pyrealsense2 as rs
import numpy as np
import cv2
import threading
from scipy.spatial.transform import Rotation as R
import time
import os

# ZED SDK import - only imported when ZED camera classes are used
# This allows get_cam_extrinsic to work without ZED SDK installed
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    sl = None
    ZED_AVAILABLE = False

def save_rgb_image(rgb_array, save_path):
    """
    保存 observation["3rd"]["rgb"] 到指定路径
    :param rgb_array: numpy array, HxWx3, RGB格式，值范围[0,255]或[0,1]
    :param save_path: str, 保存路径
    """
    # 如果是float类型且范围在[0,1]，先转为[0,255] uint8
    if rgb_array.dtype != np.uint8:
        rgb_array = (rgb_array * 255).clip(0, 255).astype(np.uint8)
    # OpenCV保存为BGR格式
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr)    


def get_cam_extrinsic(type):
    if type == "3rd_1":
        # TODO: 填充第一个第三视角相机的实际外参
        trans=np.array([0.005296609417704192, -0.3157368291363717, 0.32600708872144163])
        quat=np.array([ -0.7548033034142615, 0.3380675715030415,-0.31014318740211866,0.4688213876957924]) # x y z w
    elif type == "3rd_2":
        # TODO: 填充第二个第三视角相机的实际外参
        trans=np.array([0.08305056557124976, 0.4562288875573745,  0.31101059961302635])
        quat=np.array([ -0.366385841322638, 0.7238074691479117,  -0.532059243228617,  0.2424399812393253]) # x y z w
        # trans=np.array([0.41319647185014996, 0.671280823421573,  0.31847372597347334])
        # quat=np.array([ -0.004178189428432915, 0.8100601972017345,  -0.5859203348840233,  0.0028462131169476428]) # x y z w
    elif type == "3rd_3":
        # TODO: 填充第三个第三视角相机的实际外参
        trans=np.array([1.0177272694179353, -0.0510392240262284, 0.37670959286453537])
        quat=np.array([ -0.5972629758372091, -0.6127793052227456,0.38451047230872026,0.3463093378321404]) # x y z w
    else:
        raise ValueError("Invalid type")

    transform = np.eye(4)
    rot = R.from_quat(quat)
    transform[:3, :3] = rot.as_matrix()
    transform[:3, 3] = trans.T

    return transform
 
class ZedCam:
    def __init__(self, serial_number, resolution=None, zed_resolution="VGA"): # resolution=(480, 640), zed_resolution="HD1080" or "VGA"
        """
        Args:
            serial_number: ZED相机序列号
            resolution: 输出图像分辨率 (height, width)，如果为None则使用相机原始分辨率
            zed_resolution: ZED相机硬件分辨率，可选 "HD1080" 或 "VGA"
        """
        self.zed = sl.Camera()
        self.zed_resolution = zed_resolution

        # if serial_number==37019563:
        #     self.init_zed_37019563(serial_number=serial_number)
        # elif serial_number==34438347:
        #     self.init_zed_34438347(serial_number=serial_number)
        # elif serial_number==30519310:
        #     self.init_zed_30519310(serial_number=serial_number)
        # else:
        #     assert False
        
        
        self.init_zed_auto(serial_number=serial_number)
        if resolution:
            self.img_size = sl.Resolution()
            self.img_size.height = resolution[0]
            self.img_size.width = resolution[1]
        else:
            self.img_size = self.zed.get_camera_information().camera_configuration.resolution
        # print(self.img_size.height)
        # print(self.img_size.width)
        self.center_crop = False
        self.center_crop_size = (480, 640)  #这是必须的吗？
        


    def init_zed_37019563(self,serial_number):
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)

        # 根据参数设置分辨率
        if self.zed_resolution == "VGA":
            init_params.camera_resolution = sl.RESOLUTION.VGA
        elif self.zed_resolution == "HD1080":
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        else:
            raise ValueError(f"不支持的ZED分辨率: {self.zed_resolution}，请使用 'HD1080' 或 'VGA'")

        init_params.camera_fps = 100

        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit()
            
        # 禁用自动曝光和增益控制
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)  # 禁用自动曝光和增益控制
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC_ROI, 0)  # 禁用自动曝光ROI
        
        # 设置固定的曝光值和增益值

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 29)  # 设置固定曝光值
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 11)      # 设置固定增益值
        
        # 设置固定亮度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4) # 设置固定亮度值

        # 设置色彩饱和度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)      # 设置固定增益值
        
        #   设置对比度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)  # 设置固定对比度

        
        # 初始化50帧
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        for _ in range(50):
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT)


    def init_zed_34438347(self,serial_number):
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)

        # 根据参数设置分辨率
        if self.zed_resolution == "VGA":
            init_params.camera_resolution = sl.RESOLUTION.VGA
        elif self.zed_resolution == "HD1080":
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        else:
            raise ValueError(f"不支持的ZED分辨率: {self.zed_resolution}，请使用 'HD1080' 或 'VGA'")

        init_params.camera_fps = 100

        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit()
            
        # 禁用自动曝光和增益控制
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)  # 禁用自动曝光和增益控制
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC_ROI, 0)  # 禁用自动曝光ROI
        
        # 设置固定的曝光值和增益值

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 32)  # 设置固定曝光值
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 10)      # 设置固定增益值
        
        # 设置固定亮度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 6) # 设置固定亮度值

        # 设置色彩饱和度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)      # 设置固定增益值
        
        #   设置色彩饱和度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)  # 设置固定曝光值

        
        # 初始化50帧
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        for _ in range(50):
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT)


    def init_zed_30519310(self,serial_number):
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)

        # 根据参数设置分辨率
        if self.zed_resolution == "VGA":
            init_params.camera_resolution = sl.RESOLUTION.VGA
        elif self.zed_resolution == "HD1080":
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        else:
            raise ValueError(f"不支持的ZED分辨率: {self.zed_resolution}，请使用 'HD1080' 或 'VGA'")

        init_params.camera_fps = 100

        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit()
            
        # 禁用自动曝光和增益控制
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)  # 禁用自动曝光和增益控制
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC_ROI, 0)  # 禁用自动曝光ROI
        
        # 设置固定的曝光值和增益值

        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 25)  # 设置固定曝光值
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 1)      # 设置固定增益值
        
        # 设置固定亮度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 6) # 设置固定亮度值

        # 设置色彩饱和度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)      # 设置固定增益值
        
        #   设置对比度
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 6)  # 设置固定曝光值

        
        # 初始化50帧
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        for _ in range(50):
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT)



    def init_zed_auto(self,serial_number):
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(serial_number)

        # 根据参数设置分辨率（修复：使用 self.zed_resolution）
        if self.zed_resolution == "VGA":
            init_params.camera_resolution = sl.RESOLUTION.VGA
        elif self.zed_resolution == "HD1080":
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        else:
            raise ValueError(f"不支持的ZED分辨率: {self.zed_resolution}，请使用 'HD1080' 或 'VGA'")

        init_params.camera_fps = 30  # Set fps at 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Use ULTRA depth mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER # Use millimeter units (for depth measurements)

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit()
            
        # Init 50 frames
        image = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()
        for _ in range(50):
            # Grab an image, a RuntimeParameters object must be given to grab()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # A new image is available if grab() returns SUCCESS
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                     
    
    
    
    def capture(self):
        image = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)
        depth_map = sl.Mat(self.img_size.width, self.img_size.height, sl.MAT_TYPE.U8_C4)
        point_cloud = sl.Mat()

        while True:
            runtime_parameters = sl.RuntimeParameters()
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
                # A new image and depth is available if grab() returns SUCCESS
                self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, self.img_size) # Retrieve left image
                self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU, self.img_size) # Retrieve depth
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.img_size)
                frame_timestamp_ms = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_microseconds()
                break
            
        rgb_image = image.get_data()[..., :3] 
        depth = depth_map.get_data()
        depth[np.isnan(depth)] = 0
        depth_image_meters = depth * 0.001
        pcd = point_cloud.get_data()
        pcd[np.isnan(pcd)] = 0
        pcd = pcd[..., :3] * 0.001
        
        if self.center_crop:
            result_dict = {
                "rgb": self.center_crop_img(rgb_image),
                "depth": self.center_crop_img(depth_image_meters),
                "pcd": self.center_crop_img(pcd),
                "timestamp_ms": frame_timestamp_ms / 1000.0,
            }
        else:
            result_dict = {
                "rgb": rgb_image,
                "depth": depth_image_meters,
                "pcd": pcd,
                "timestamp_ms": frame_timestamp_ms / 1000.0,
            }
        return result_dict
    
    

    def center_crop_img(self, img):
        if len(img.shape) == 2:
            crop_img = np.zeros((self.center_crop_size[0], self.center_crop_size[1]), dtype=img.dtype)
            crop_img = img[(img.shape[0] - self.center_crop_size[0]) // 2: (img.shape[0] + self.center_crop_size[0]) // 2,
                          (img.shape[1] - self.center_crop_size[1]) // 2: (img.shape[1] + self.center_crop_size[1]) // 2]
            return crop_img
        else:
            channel = img.shape[-1]
            crop_img = np.zeros((self.center_crop_size[0], self.center_crop_size[1], channel), dtype=img.dtype)
            crop_img = img[(img.shape[0] - self.center_crop_size[0]) // 2: (img.shape[0] + self.center_crop_size[0]) // 2,
                            (img.shape[1] - self.center_crop_size[1]) // 2: (img.shape[1] + self.center_crop_size[1]) // 2]
        return crop_img
        
    
    def stop(self):
        # Close the camera
        self.zed.close()
        

class Camera:
    def __init__(self, camera_type="all", timestamp_tolerance_ms=80, zed_resolution="VGA"):
        """
        Args:
            camera_type: 相机类型，可选 "all", "3rd_1", "3rd_2", "3rd_3"
            timestamp_tolerance_ms: 时间戳容差（毫秒）
            zed_resolution: ZED相机硬件分辨率，可选 "HD1080" 或 "VGA"
        """
        # 修改为实际连接的ZED相机序列号
        static_serial_number_1 = 30519310  # 当前连接的ZED 2i相机
        static_serial_number_2 = 30519310  # 备用（如果只有一个相机，使用同一个序列号）
        static_serial_number_3 = 30519310  # 备用（如果只有一个相机，使用同一个序列号）

        if camera_type == "all":
            # 三个第三视角相机
            self.cams =  [
                ZedCam(serial_number=static_serial_number_1, zed_resolution=zed_resolution),
                ZedCam(serial_number=static_serial_number_2, zed_resolution=zed_resolution),
                ZedCam(serial_number=static_serial_number_3, zed_resolution=zed_resolution)
            ]
            self.camera_types = ["3rd_1", "3rd_2", "3rd_3"]

        elif camera_type == "3rd_1":
            self.cams = [ZedCam(serial_number=static_serial_number_1, zed_resolution=zed_resolution)]
            self.camera_types = ["3rd_1"]

        elif camera_type == "3rd_2":
            self.cams = [ZedCam(serial_number=static_serial_number_2, zed_resolution=zed_resolution)]
            self.camera_types = ["3rd_2"]

        elif camera_type == "3rd_3":
            self.cams = [ZedCam(serial_number=static_serial_number_3, zed_resolution=zed_resolution)]
            self.camera_types = ["3rd_3"]

        else:
            raise ValueError("Invalid camera type, please choose from 'all', '3rd_1', '3rd_2', '3rd_3'")

        self.timestamp_tolerance_ms = timestamp_tolerance_ms
        
        
    def _capture_frame(self, idx, result_dict, start_barrier, done_barrier):
        """
        start_barrier: A threading.Barrier to ensure all threads start capturing at the same time.
        done_barrier: A threading.Barrier to ensure all threads finish capturing before main thread proceeds.
        """
        cam = self.cams[idx]
        camera_type = self.camera_types[idx]
        # Wait here until all threads are ready (software-level synchronization)
        start_barrier.wait()
        result_dict[camera_type] = cam.capture()
        # Signal that this thread is done
        done_barrier.wait()
        
    def capture_frames_multi_thread(self):
        result_dict = {}
        if len(self.cams) == 1:
            result_dict[self.camera_types[0]] = self.cams[0].capture()
            _ = [result_dict[cam].pop("timestamp_ms", None) for cam in result_dict] # remove timestamps
            return result_dict
        
        else:
            num_cameras = len(self.cams)

            # Two barriers: one to synchronize the start, one to wait for all threads to finish
            start_barrier = threading.Barrier(num_cameras)
            done_barrier = threading.Barrier(num_cameras)

            threads = []

            for idx in range(num_cameras):
                t = threading.Thread(
                    target=self._capture_frame,
                    args=(idx, result_dict, start_barrier, done_barrier)
                )
                threads.append(t)
                t.start()

            # Wait for all threads to finish
            for t in threads:
                t.join()

            # -------------------------
            # Timestamp alignment step
            # -------------------------
            # 1) Gather all timestamps
            timestamps = [result_dict[cam]["timestamp_ms"] for cam in result_dict]
            _ = [result_dict[cam].pop("timestamp_ms", None) for cam in result_dict] # remove timestamps
            
            # 2) Compute min, max, and check difference
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            diff_ts = max_ts - min_ts  # in ms

            # 3) Compare difference with the tolerance
            if diff_ts > self.timestamp_tolerance_ms:
                print("Timestamps are not aligned, difference is", diff_ts, "ms,", "discard frames")
                return None
            else:
                return result_dict
    
    
    def capture(self):
        while True:
            result_dict = self.capture_frames_multi_thread()
            if result_dict is not None:
                break
        return result_dict
    
    
    def stop(self):
        for cam in self.cams:
            cam.stop()



           
            
if __name__ == "__main__":
    # 测试所有三个相机
    test_camera_type = "3rd_1"  # 使用所有三个相机

    cameras = Camera(camera_type=test_camera_type)

    import open3d as o3d
    observation = cameras.capture()

    # 为什么一定要这样写，为什么不可以直接将observation["3rd"]["rgb"][:,:,::-1].copy()传给is_pcd
    # 因为这种逆序操作会破坏连续性，后续如果使用reshape这类操作时会产生错位的数据
    def convert_pcd_to_base(type, pcd):
        transform = get_cam_extrinsic(type)

        h, w = pcd.shape[:2]
        pcd = pcd.reshape(-1, 3)

        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
        # pcd = (np.linalg.inv(transform) @ pcd.T).T[:, :3]
        pcd = (transform @ pcd.T).T[:, :3]

        pcd = pcd.reshape(h, w, 3)
        return pcd

    def filter_pcd_by_range(pcd, rgb, x_range=(0, 1.0), y_range=(-1, 1), z_range=(0, 1)):
        """
        根据空间范围过滤点云
        :param pcd: numpy array, 点云数据 (H, W, 3)
        :param rgb: numpy array, RGB颜色数据 (H, W, 3)
        :param x_range: x坐标范围
        :param y_range: y坐标范围
        :param z_range: z坐标范围
        :return: 过滤后的点云和颜色 (N, 3)
        """
        # 展平点云和颜色
        pcd_flat = pcd.reshape(-1, 3)
        rgb_flat = rgb.reshape(-1, 3)

        # 创建范围掩码
        mask = (
            (pcd_flat[:, 0] >= x_range[0]) & (pcd_flat[:, 0] <= x_range[1]) &
            (pcd_flat[:, 1] >= y_range[0]) & (pcd_flat[:, 1] <= y_range[1]) &
            (pcd_flat[:, 2] >= z_range[0]) & (pcd_flat[:, 2] <= z_range[1])
        )

        # 应用掩码
        filtered_pcd = pcd_flat[mask]
        filtered_rgb = rgb_flat[mask]

        return filtered_pcd, filtered_rgb, mask

    def vis_pcd(pcd, rgb, title="Point Cloud", apply_filter=False):
        """
        可视化点云和坐标系
        :param pcd: numpy array, 点云数据
        :param rgb: numpy array, RGB颜色数据
        :param title: 窗口标题
        :param apply_filter: 是否应用范围过滤
        """
        if apply_filter:
            # 过滤点云
            pcd_flat, rgb_flat, mask = filter_pcd_by_range(pcd, rgb,
                                                            x_range=(0, 1.0),
                                                            y_range=(-1, 1),
                                                            z_range=(-0.05, 1))
            rgb_flat = rgb_flat / 255.0
            print(f"    过滤后保留点数: {pcd_flat.shape[0]} (原始: {pcd.shape[0]*pcd.shape[1]})")
        else:
            # 将点云和颜色转换为二维的形状 (N, 3)
            pcd_flat = pcd.reshape(-1, 3)  # (H*W, 3)
            rgb_flat = rgb.reshape(-1, 3) / 255.0  # (H*W, 3)

        # 创建点云对象
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_flat)
        pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_flat)

        # 创建坐标系
        # 参数说明：
        # size: 坐标系的大小
        # origin: 坐标系的原点位置
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,  # 坐标系的大小，可以根据你的点云尺度调整
            origin=[0, 0, 0]  # 坐标系的原点位置
        )  # 用来判断标定是否正确

        # 可视化点云和坐标系
        print(f"Visualizing {title}...")
        o3d.visualization.draw_geometries([pcd_o3d, coordinate_frame], window_name=title)

    # 第一步：处理并分别可视化每个相机的点云（基坐标系，应用范围过滤）
    print("=" * 60)
    print("第一步：分别可视化三个相机的点云（基坐标系，范围过滤）")
    print("=" * 60)

    for cam_type in observation.keys():
        print(f"\n处理 {cam_type} 相机数据...")

        # 转换RGB格式（BGR -> RGB）
        observation[cam_type]["rgb"] = observation[cam_type]["rgb"][:,:,::-1].copy()

        # 转换点云到基坐标系
        observation[cam_type]["pcd"] = convert_pcd_to_base(cam_type, observation[cam_type]["pcd"])

        # 打印转换后点云的统计信息
        pcd_after = observation[cam_type]["pcd"]
        print(f"  基坐标系下点云统计:")
        print(f"    形状: {pcd_after.shape}")
        print(f"    X范围: [{pcd_after[..., 0].min():.3f}, {pcd_after[..., 0].max():.3f}]")
        print(f"    Y范围: [{pcd_after[..., 1].min():.3f}, {pcd_after[..., 1].max():.3f}]")
        print(f"    Z范围: [{pcd_after[..., 2].min():.3f}, {pcd_after[..., 2].max():.3f}]")

        # 可视化基坐标系下的点云（应用范围过滤）
        vis_pcd(observation[cam_type]["pcd"], observation[cam_type]["rgb"],
                title=f"{cam_type} Point Cloud (Base Frame, Filtered)", apply_filter=True)

        # 保存RGB图像
        save_path = f"/media/casia/data4/lpy/debug_{cam_type}.png"
        save_rgb_image(observation[cam_type]["rgb"], save_path)
        print(f"图像已保存到: {save_path}")

    # 第二步：合并三个相机的点云并可视化（应用范围过滤）
    # print("\n" + "=" * 60)
    # print("第二步：合并三个相机的点云并可视化（范围过滤）")
    # print("=" * 60)

    # # 收集所有相机的点云和RGB数据，并应用范围过滤
    # merged_pcd_list = []
    # merged_rgb_list = []

    # for cam_type in ["3rd_1", "3rd_2", "3rd_3"]:
    # # for cam_type in ["3rd_1", "3rd_2"]:
    #     # 应用范围过滤
    #     pcd_filtered, rgb_filtered, mask = filter_pcd_by_range(
    #         observation[cam_type]["pcd"],
    #         observation[cam_type]["rgb"],
    #         x_range=(-0.1, 0.9),
    #         y_range=(-0.5, 0.5),
    #         z_range=(-0.1, 0.9)
    #     )

    #     merged_pcd_list.append(pcd_filtered)
    #     merged_rgb_list.append(rgb_filtered)

    #     print(f"{cam_type} 过滤后点云形状: {pcd_filtered.shape}")

    # # 拼接所有点云
    # merged_pcd = np.concatenate(merged_pcd_list, axis=0)  # (N_total, 3)
    # merged_rgb = np.concatenate(merged_rgb_list, axis=0)  # (N_total, 3)

    # print(f"\n合并后点云形状: {merged_pcd.shape}")
    # print(f"合并后RGB形状: {merged_rgb.shape}")

    # # 可视化合并后的点云
    # merged_rgb_normalized = merged_rgb / 255.0

    # pcd_o3d = o3d.geometry.PointCloud()
    # pcd_o3d.points = o3d.utility.Vector3dVector(merged_pcd)
    # pcd_o3d.colors = o3d.utility.Vector3dVector(merged_rgb_normalized)

    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.1,
    #     origin=[0, 0, 0]
    # )

    # print("\n可视化合并后的点云...")
    # o3d.visualization.draw_geometries([pcd_o3d, coordinate_frame], window_name="Merged Point Cloud (3 Cameras, Filtered)")

    # cameras.stop()
    # print("\n测试完成！")
