import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import cv2


class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False, exposure=40000):
        self.name = name
        assert serial_number in self.get_device_serial_numbers(), "Serial number not found"
        self.serial_number = serial_number
        self.depth = depth

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)

        # —— 启动相机 —— #
        self.profile = self.pipe.start(self.cfg)

        # —— 锁定 RGB 传感器，关闭自动曝光并设置手动曝光 —— #
        self.color_sensor = None
        for s in self.profile.get_device().query_sensors():
            try:
                if s.get_info(rs.camera_info.name) in ("RGB Camera", "Color Sensor"):
                    self.color_sensor = s
                    break
            except Exception:
                pass
        # 若未找到，退回第一个传感器（不推荐，但能避免崩溃）
        if self.color_sensor is None:
            self.color_sensor = self.profile.get_device().query_sensors()[0]

        # 关闭自动曝光
        if self.color_sensor.supports(rs.option.enable_auto_exposure):
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)

        # 设置手动曝光（微秒），并做范围裁剪
        if self.color_sensor.supports(rs.option.exposure):
            rng = self.color_sensor.get_option_range(rs.option.exposure)
            exp_val = float(min(max(exposure, rng.min), rng.max))
            self.color_sensor.set_option(rs.option.exposure, exp_val)

        # 可选：根据供电频率减轻灯光条纹（50/60Hz）
        if self.color_sensor.supports(rs.option.power_line_frequency):
            try:
                self.color_sensor.set_option(rs.option.power_line_frequency, rs.power_line_frequency.freq_50hz)
            except Exception:
                pass

        # —— 深度对齐，仅在开启深度时创建 —— #
        self.align = rs.align(rs.stream.color) if self.depth else None

    def read(self):
        frames = self.pipe.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        if self.depth:
            depth_frame = frames.get_depth_frame()
        else:
            depth_frame = None

        if color_frame and color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        try:
            self.pipe.stop()
        finally:
            try:
                self.cfg.disable_all_streams()
            except Exception:
                pass


if __name__ == '__main__':
    # 注意：彩色流常见不支持 90fps，如报 Couldn't resolve requests，请改成 30 或 60。
    cam = RSCapture(name="front", serial_number='323622271380', fps=90, exposure=15000)
    ok, frame = cam.read()
    if ok:
        cv2.imshow("Photo(BGR)", frame if frame.ndim == 3 else frame[..., :3])
        cv2.imwrite("photo.jpg", frame if frame.ndim == 3 else frame[..., :3])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未找到图像")

    cam.close()
