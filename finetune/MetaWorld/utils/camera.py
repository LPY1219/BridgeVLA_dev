# reference implementation: https://github.com/mattcorsaro1/mj_pc
# with personal modifications


import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d



"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""
def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

def sample_from_mask(mask, num_samples=100):
    on = np.array(mask.nonzero()[::-1]).T.astype(np.float64)  # Reverse the order to get (x, y)
    if len(on) == 0:
        on = np.array((mask == 0).nonzero()[::-1]).T.astype(np.float64)  # Same for the empty case
    sample_ind = np.random.choice(len(on), num_samples, replace=True)
    samples = on[sample_ind]
    return samples

def get_grasp(seg, depth, camera_to_world_transform, r=5):
    samples = sample_from_mask(seg, 500)
    def loss(i):
        return np.linalg.norm(samples - samples[i], axis=1).sum()
    grasp_2d = samples[np.argmin([loss(i) for i in range(len(samples))])]
    neighbor_threshold = r
    neighbors = samples[np.linalg.norm(samples - grasp_2d, axis=1) < neighbor_threshold]
    # neighbors_d = np.array([[sample_with_binear(depth, kp)] for kp in neighbors])
    neighbors_d = np.array([[sample_with_binear(depth, kp)] for kp in neighbors])
    d = np.median(neighbors_d)

    return transform_pixels_to_world(grasp_2d, [d], camera_to_world_transform)


def sample_with_binear(fmap, kp):
    max_x, max_y = fmap.shape[1]-1, fmap.shape[0]-1
    x0, y0 = max(0, int(kp[0])), max(0, int(kp[1]))
    x1, y1 = min(max_x, x0+1), min(max_y, y0+1)
    x, y = max(0, kp[0]-x0), max(0, kp[1]-y0)
    fmap_x0y0 = fmap[y0, x0]
    fmap_x1y0 = fmap[y0, x1]
    fmap_x0y1 = fmap[y1, x0]
    fmap_x1y1 = fmap[y1, x1]
    fmap_y0 = fmap_x0y0 * (1-x) + fmap_x1y0 * x
    fmap_y1 = fmap_x0y1 * (1-x) + fmap_x1y1 * x
    feature = fmap_y0 * (1-y) + fmap_y1 * y
    return feature

def transform_pixels_to_world(pixels, depth, camera_to_world_transform):
    # pixels in uv space (not row col)
    # sample from the depth map using the pixel locations with bilinear sampling
    pixels = pixels.astype(float)
    depth = np.array(depth)

    # form 4D homogenous camera vector to transform - [x * z, y * z, z, 1]
    # cam_pts = [pixels[..., 1:2] * depth, pixels[..., 0:1] * depth, depth, np.ones_like(depth)]
    cam_pts = [pixels[..., 0:1] * depth, pixels[..., 1:2] * depth, depth, np.ones_like(depth)]
    cam_pts = np.concatenate(cam_pts, axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do camera to robot frame transform
    mat_reshape = [1] * len(cam_pts.shape[:-1]) + [4, 4]
    cam_trans = camera_to_world_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    points = np.matmul(cam_trans, cam_pts[..., None])[..., 0]  # shape [..., 4]
    return points[..., :3]

# 
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""
class Camera(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, sim, cam_names:List, img_size=84):
        super(Camera, self).__init__()

        self.sim = sim

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        # Filter out cameras that don't exist in the environment
        available_cameras = list(self.sim.model.camera_names) if hasattr(self.sim.model, 'camera_names') else []
        valid_cam_names = []
        self.cam_ids = []  # Store camera IDs for each valid camera
        self.cam_mats = []
        
        self.env_all_cameras = ['topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV']
        
        for cam_name in cam_names:
            # Try to get camera ID - camera_name2id will raise ValueError if camera doesn't exist
            try:
                cam_id = self.sim.model.camera_name2id(cam_name)
                # Validate camera ID is within valid range
                if cam_id < 0 or cam_id >= len(self.sim.model.cam_fovy):
                    print(f"Warning: Camera '{cam_name}' has invalid ID {cam_id}, skipping.")
                    continue
                
                fovy = math.radians(self.sim.model.cam_fovy[cam_id])
                f = self.img_height / (2 * math.tan(fovy / 2))
                cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
                
                valid_cam_names.append(cam_name)
                self.cam_ids.append(cam_id)
                self.cam_mats.append(cam_mat)
            except (ValueError, KeyError, IndexError, AttributeError) as e:
                print(f"Warning: Camera '{cam_name}' not available, skipping. Error: {e}")
                continue
        
        if len(valid_cam_names) == 0:
            raise ValueError(f"No valid cameras found. Requested: {cam_names}, Available: {available_cameras}")
        
        self.cam_names = valid_cam_names
    
    def get_camera_extrinsic_matrix(self, cam_id):
        """
        Get camera-to-world transformation matrix.
        @param cam_id: Camera ID (not body ID) from sim.model.camera_name2id()
        """
        # cam_id is the camera ID, use it directly to access camera data
        camera_pos = self.sim.data.cam_xpos[cam_id]
        camera_rot = self.sim.data.cam_xmat[cam_id].reshape(3, 3)
        
        R = np.zeros((4, 4))
        R[:3, :3] = camera_rot
        R[:3, 3] = camera_pos
        R[3, 3] = 1.0
        
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        R = R @ camera_axis_correction
        return R
    
    def get_camera_intrinsic_matrix(self, cam_id):
        
        fovy = math.radians(self.sim.model.cam_fovy[cam_id])

        f = self.img_height / (2 * math.tan(fovy / 2))
        K = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
        return K

    def get_camera_transform_matrix(self, cam_id):
        R = self.get_camera_extrinsic_matrix(cam_id)
        K = self.get_camera_intrinsic_matrix(cam_id)
        K_exp = np.eye(4)
        K_exp[:3, :3] = K

        # Takes a point in world, transforms to camera frame, and then projects onto image plane.
        inv_R = np.zeros((4, 4))
        inv_R[:3, :3] = R[:3, :3].T
        inv_R[:3, 3] = -inv_R[:3, :3].dot(R[:3, 3])
        inv_R[3, 3] = 1.0
        
        return K_exp @ inv_R
    
    def captureSeg(self, camera_name, device_id=0):
        seg = self.sim.render(self.img_width, self.img_height, camera_name=camera_name, segmentation=True, device_id=device_id)
        seg = self.verticalFlip(seg)
        return seg
    
    def captureDepth(self, camera_name, device_id=0):
        rgb, depth = self.sim.render(width=self.img_width, height=self.img_height, camera_name=camera_name, depth=True, device_id=device_id)
        depth_convert = self.depthimg2Meters(depth)
        depth_convert = self.verticalFlip(depth_convert)
        return depth_convert
        
    def capturePointCloud(self, save_img_dir=None, device_id=0):
        o3d_clouds = []
        color_imgs = []
        cam_poses = []
        depths = []
        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            color_img, depth = self.captureImage(self.cam_names[cam_i], capture_depth=True, device_id=device_id)
            depths.append(depth)
            color_imgs.append(color_img)
            # If directory was provided, save color and depth images
            #    (overwriting previous)
            if save_img_dir != None:
                self.saveImg(depth, save_img_dir, "depth_test_" + str(cam_i))
                self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))

            # convert camera matrix and depth image to Open3D format, then
            #    generate point cloud
            
            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth)
            
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
            
            cam_body_id = self.env_all_cameras.index(self.cam_names[cam_i])
            c2w = self.get_camera_extrinsic_matrix(cam_body_id)
            transformed_cloud = o3d_cloud.transform(c2w)
            o3d_clouds.append(transformed_cloud)

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        # get numpy array of point cloud, (position, color)
        combined_cloud_points = np.asarray(combined_cloud.points)
        # color is automatically normalized to [0,1] by open3d
        
        combined_cloud_colors = np.concatenate([color_img.reshape(-1, 3) for color_img in color_imgs], axis=0) # range [0, 255]
        combined_cloud = np.concatenate((combined_cloud_points, combined_cloud_colors), axis=1)
        depths = np.array(depths).squeeze()
        return combined_cloud, depths

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, camera_name, capture_depth=True, device_id=0):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=camera_name, depth=capture_depth, device_id=device_id)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)

            depth_convert = self.depthimg2Meters(depth)
            img = self.verticalFlip(img)
            return img, depth_convert
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img/img.max()*255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + '/' + filename + ".jpg")
