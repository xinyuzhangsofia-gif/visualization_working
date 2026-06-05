import yaml
import path_setup
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation
from scipy.io import loadmat

#from center point to 8 corners
def boxes_to_corners_3d(boxes):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    template = torch.tensor([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ],dtype=torch.float32,device=boxes.device) / 2

    lidar_corners=boxes[:, None, 3:6] * template[None, :, :]  # lwh*template (N,1,3)*(1,8,3) = (N,8,3)

    yaw=boxes[:, 6] 
    c=torch.cos(yaw)
    s=torch.sin(yaw)

    R=torch.zeros((boxes.shape[0], 3, 3), device=boxes.device) #(N,3,3),get the rotation matrix for each box
    R[:, 0, 0]=c
    R[:, 0, 1]=-s
    R[:, 1, 0]=s
    R[:, 1, 1]=c
    R[:, 2, 2]=1    

    lidar_corners=torch.matmul(lidar_corners, R.transpose(1, 2))  # (N, 8, 3)

    lidar_corners+=boxes[:, None, 0:3]  # (N, 8, 3)
    return lidar_corners


def transform_lidar_to_radar(lidar_corners:torch.Tensor,R,T) :
    radar_corners = torch.matmul(lidar_corners, R.T) + T
    return radar_corners


def cartesian_to_rae(radar_corners):
    x = radar_corners[..., 0] # x, y, z are the last dimension of lidar_corners
    y = radar_corners[..., 1]
    z = radar_corners[..., 2]
    r_xy = torch.sqrt(x**2 + y**2) 
    r = torch.sqrt(x**2 + y**2 + z**2) 
    azimuth = torch.atan2(-y, x)
    azimuth = torch.rad2deg(azimuth)
    elevation = torch.atan2(z, r_xy+1e-6)  # Add small epsilon to avoid division by zero
    rae_corners=torch.stack((r, azimuth, elevation), dim=-1)
    rae_corners=rae_corners.cpu().numpy()
    return rae_corners

def get_ra_bbx_2d(rae_corners):
    
    num_boxes = rae_corners.shape[0]
    bbxes_2d = np.zeros((num_boxes, 5, 2), dtype=np.float32)

    for i in range(num_boxes):
        ra_points = rae_corners[i][:, [0, 1]]

        r_vals = ra_points[:, 0]
        a_vals = ra_points[:, 1]

        r_min = np.min(r_vals)
        r_max = np.max(r_vals)
        a_min = np.min(a_vals)
        a_max = np.max(a_vals)

        bbx_2d = np.asarray([
            [a_min, r_min],
            [a_max, r_min],
            [a_max, r_max],
            [a_min, r_max],
            [a_min, r_min]
        ], dtype=np.float32)
        bbxes_2d[i] = bbx_2d

    return bbxes_2d


def load_axis_from_mat(info_array_path):

    mat_data = loadmat(info_array_path)

    arr_range= mat_data['arrRange'][0]
    arr_azimuth_deg = mat_data['arrAzimuth'][0]
    arr_elevation_deg = mat_data['arrElevation'][0]

    return arr_range, arr_azimuth_deg, arr_elevation_deg


def load_lidar2radar_calib(yml_path):
    with open(yml_path, "r") as f:
        data = yaml.safe_load(f)

    R = torch.tensor(data["calib_lidar2radar"]["R"], dtype=torch.float32)
    T = torch.tensor(data["calib_lidar2radar"]["T"], dtype=torch.float32)

    return R, T


def load_full_camera_calib(path_calib):
    with open(path_calib, "r") as f:
        data = yaml.safe_load(f)

     # intrinsic matrix K
    K = np.array([
        [data["fx"], 0.0, data["px"]],
        [0.0, data["fy"], data["py"]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    

    # distortion coefficients
    distortion = np.array([data["k1"],data["k2"],data["k3"],data["k4"],data["k5"]],
                          dtype=np.float64).reshape(-1, 1)

    # extrinsic: lidar -> camera
    yaw = data["yaw_ldr2cam"]
    pitch = data["pitch_ldr2cam"]
    roll = data["roll_ldr2cam"]

    R = Rotation.from_euler('zyx',[yaw, pitch, roll],degrees=True).as_matrix()

    T= np.array([data["x_ldr2cam"],data["y_ldr2cam"],data["z_ldr2cam"]],
                dtype=np.float64).reshape(3, 1)

    return K, distortion,R, T


def transform_lidar_to_camera(lidar_corners:torch.Tensor,T ,R):

        rot_default = np.array([
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0]
        ])
        r_rotation_default = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        T=np.array(T,dtype=np.float32).reshape(3)  # (3,)

        R = R @ rot_default.T

        Tr = np.eye(4, dtype=np.float32)
        Tr[:3, :3] = R
        Tr[:3, 3] = T

        original_shape = lidar_corners.shape[:-1]

        lidar_corners=lidar_corners.reshape(-1,3)

        points_hom = np.vstack([lidar_corners.T,np.ones((1, lidar_corners.shape[0]))])                                          # (4, N*8)

        camera_corners = Tr @ r_rotation_default@points_hom

        
        camera_corners = camera_corners[:3, :].T
        camera_corners = torch.tensor(camera_corners.reshape(*original_shape, 3))
    
        return camera_corners


def undistort_image(camera_img,K, distortion):
    h, w = camera_img.shape[:2]
    img_size = (w, h)
    ncm, _ = cv2.getOptimalNewCameraMatrix(K, distortion, img_size, alpha=0.0)
    for j in range(3):
        for i in range(3):
            K[j,i] = ncm[j, i]
    r_cam = np.array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, distortion, r_cam, K, img_size,cv2.CV_32FC1
    )
    img_undistort = cv2.remap(camera_img, map_x, map_y, cv2.INTER_LINEAR)

    return img_undistort


def camera_corners_to_2d_undistort(camera_corners,K): 
    camera_corners = camera_corners.cpu().numpy()

    x = camera_corners[..., 0]
    y = camera_corners[..., 1]
    z = camera_corners[..., 2]

    # valid only if the 3D point is finite and sufficiently in front of the camera
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 1e-6)

    u = np.full_like(x, float('nan'))
    v = np.full_like(y, float('nan'))

    if valid_mask.any():
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]

        # normalized image coordinates
        xn = x_valid / z_valid
        yn = y_valid / z_valid

        #directly use normalized coordinates
        u[valid_mask] = K[0,0] * xn + K[0,2]
        v[valid_mask] = K[1,1] * yn + K[1,2]

    camera_2d_points_undistort = np.stack([u, v], axis =-1)
    return camera_2d_points_undistort, valid_mask


def get_ra_cartesian_limits(arr_range, arr_azimuth_deg):
    r_max = arr_range.max()

    a_min = np.deg2rad(arr_azimuth_deg.min())
    a_max = np.deg2rad(arr_azimuth_deg.max())

    x_min = r_max * np.sin(a_min)
    x_max = r_max * np.sin(a_max)

    y_min = 0
    y_max = r_max * np.cos(0)

    return x_min, x_max, y_min, y_max
