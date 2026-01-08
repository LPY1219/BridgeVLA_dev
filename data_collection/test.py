import pyzed.sl as sl

zed = sl.Camera()
init_params = sl.InitParameters()

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Open failed:", err)
    exit(1)

serial = zed.get_camera_information().serial_number
print("ZED serial number:", serial)

zed.close()