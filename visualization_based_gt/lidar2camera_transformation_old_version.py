import os
import path_setup

os.environ.pop("QT_QPA_PLATFORM", None)
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ["MPLBACKEND"] = "TkAgg"

import matplotlib
matplotlib.use("TkAgg")
from lidar2radar_transformation import read_info_label,boxes_to_corners_3d
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
import os
import cv2
import numpy as np
import yaml

def get_lidar_path_os2(pcd_dir,os2_64_idx):
    for fname in sorted(os.listdir(pcd_dir)):
        if fname.startswith(f"os2-64_{os2_64_idx}"):
            return os.path.join(pcd_dir,fname)
    raise FileNotFoundError(f"os2-64-lidar file not found for idx{os2_64_idx} in {pcd_dir}")

def get_lidar_path_os1(pcd_dir,os1_128_idx):
    for fname in sorted(os.listdir(pcd_dir)):
        if fname.startswith(f"os1-128_{os1_128_idx}"):
            return os.path.join(pcd_dir,fname)
    raise FileNotFoundError(f"os1-128-lidar file not found for idx{os1_128_idx} in {pcd_dir}")


def get_camera_path(camera_dir,cam_front_idx):
    for fname in sorted(os.listdir(camera_dir)):
        if fname.startswith(f"cam-front_{cam_front_idx}"):
            return os.path.join(camera_dir,fname)

    raise FileNotFoundError(f"cam-front file not found for idx{cam_front_idx} in {camera_dir}")


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


def crop_lidar_points(pcd_np):
    x = pcd_np[:,0]
    y = pcd_np[:,1]
    z = pcd_np[:,2]
    mask = ((x > 0) & (x < 80) &
           (y > -500) & (y < 500) &
           (z > -0.9) & (z < 100))
    
    return pcd_np[mask], mask


def transform_lidar_to_camera(lidar_corners:torch.Tensor,T ,R, calib_seq):

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
        
        #R = Rotation.from_euler('zyx',[yaw, pitch, roll],degrees=True).as_matrix()
        R = R @ rot_default.T

        Tr = np.eye(4, dtype=np.float32)
        Tr[:3, :3] = R
        Tr[:3, 3] = T

        original_shape = lidar_corners.shape[:-1]

        LidarToCamera = np.array([[ 0.9998872963,  0.0087265355,  0.0122165356,  0.1],
                                  [-0.0087258842,  0.9999619231, -0.0001066121,  0.0],
                                  [-0.0122170008,  0.0000000000,  0.9999253697, -0.7],
                                  [ 0.0,           0.0,           0.0,           1.0]
                                ])
        lidar_corners=lidar_corners.reshape(-1,3)

        points_hom = np.vstack([lidar_corners.T,np.ones((1, lidar_corners.shape[0]))])                                          # (4, N*8)
        if calib_seq==2:
            camera_corners = LidarToCamera @ r_rotation_default@points_hom
            
        else:
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


def camera_corners_to_2d_with_distortion(camera_corners, K, distortion):
    camera_corners = camera_corners.cpu().numpy()
    original_shape = camera_corners.shape[:-1]
    pts_3d = camera_corners.reshape(-1, 3).astype(np.float64)  # N*8,3

    valid_mask = np.isfinite(pts_3d).all(axis=1) & (pts_3d[:, 2] > 1e-6)

    pts_2d = np.full((pts_3d.shape[0], 2), np.nan, dtype=np.float64)
    #use distortion way to find 2d point
    if valid_mask.any():
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        K = np.asarray(K, dtype=np.float64)
        distortion = np.asarray(distortion, dtype=np.float64).reshape(-1,1)
        pts_3d = pts_3d[valid_mask].reshape(-1, 1, 3)  #N*8,1,3

        proj, _ = cv2.projectPoints(
            pts_3d,
            rvec,
            tvec,
            K,
            distortion
        )
        pts_2d[valid_mask] = proj.reshape(-1, 2)
        pts_2d = pts_2d.reshape(*original_shape,2)
        valid_mask = valid_mask.reshape(*original_shape)
    

    return pts_2d, valid_mask


def camera_corners_to_2d_no_distortion(camera_corners,K_new):  #without distortion
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
        u[valid_mask] = K_new[0,0] * xn + K_new[0,2]
        v[valid_mask] = K_new[1,1] * yn + K_new[1,2]

    camera_2d_points_undistort = np.stack([u, v], axis =-1)
    return camera_2d_points_undistort, valid_mask


def visualize_bbx_on_camera(camera_2d_points, image,valid_mask):
    #camera_2d_points = camera_2d_points.detach().cpu().numpy()
    #valid_mask=valid_mask.detach().cpu().numpy()

    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    image_with_bbx=image.copy()
    h, w = image.shape[:2]

    for i in range(camera_2d_points.shape[0]):
        box = camera_2d_points[i]
        box_valid = valid_mask[i]

        for start, end in edges:
            # skip this edge if either endpoint is invalid in 3D
            if not (box_valid[start] and box_valid[end]):
                continue

            point1 = box[start]
            point2 = box[end]

            # skip this edge if the projected 2D points are not finite
            if not (np.isfinite(point1).all() and np.isfinite(point2).all()):
                continue

            # optional: skip points that are far outside the image
            if not (-1000 < point1[0] < w + 1000 and -1000 < point1[1] < h + 1000):
                continue
            if not (-1000 < point2[0] < w + 1000 and -1000 < point2[1] < h + 1000):
                continue

            x1, y1 = np.round(point1).astype(np.int32)
            x2, y2 = np.round(point2).astype(np.int32)

            cv2.line(image_with_bbx, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image_with_bbx


def draw_pcd_on_camera(camera_points,image_with_bbx, intrinsics, point_size=1):
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

    image_with_pcd = image_with_bbx.copy()
    h, w = image_with_pcd.shape[:2]

    x = camera_points[:, 0]
    y = camera_points[:, 1]
    z = camera_points[:, 2]
    way = 1


    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z) & (z > 1e-6)
    if way == 1:
        if valid.any():
            x = x[valid]
            y = y[valid]
            z = z[valid]

            u = intrinsics[0, 0] * (x / z) + intrinsics[0, 2]
            v = intrinsics[1, 1] * (y / z) + intrinsics[1, 2]

            uvz = torch.stack([u, v, z], dim=-1).cpu().numpy()

            #rely on the depth to draw
            depth = uvz[:, 2]

            depth_min = 0.0
            depth_max = 80.0
            depth = np.clip(depth, depth_min, depth_max)

            depth_norm = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            #depth_norm = 255 - depth_norm 


            for (px, py, _), d in zip(uvz, depth_norm):
                px = int(round(px))
                py = int(round(py))

                if 0 <= px < w and 0 <= py < h:
                    color = cv2.applyColorMap(
                        np.array([[d]], dtype=np.uint8),
                        cv2.COLORMAP_RAINBOW
                    )[0, 0]
                    color = (color * 0.5).astype(np.uint8)

                    color = tuple(int(c) for c in color)
                    cv2.circle(image_with_pcd, (px, py), point_size, color, -1)

    return image_with_pcd
    

if __name__ == "__main__":

    #data_choose
    frame_idx = 0
    sequence=16

    choose_info_label='info_label_rev2'  # label_version 0:rev2, 1:info_label
    choose_camera=1      #1:camera 1 left side, 2:camera 2:right side
    choose_lidar=1       #0:os1-128 ,1:os2-64
    calib_seq = 1     #0:calib_seq, 1:calib_seqv2, 2:calib_init
    #data_path
    label_dir = f'/home/local/xinyu/KRadar/{sequence}/{choose_info_label}'
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    if choose_lidar==0:
        pcd_dir = f'/home/local/xinyu/KRadar/{sequence}/os1-128'
    elif choose_lidar==1:
        pcd_dir = f'/home/local/xinyu/KRadar/{sequence}/os2-64'

    if choose_camera==1:
        camera_dir = f'/home/local/xinyu/KRadar/{sequence}/cam_1_front'
    elif choose_camera==2:
        camera_dir=f'/home/local/xinyu/KRadar/{sequence}/cam_2_front'

    if calib_seq==0:
        if sequence<10:
            path_calib = f'/home/local/xinyu/KRadar/calib_seq/seq_0{sequence}/cam_{choose_camera}.yml'
        else:
            path_calib = f'/home/local/xinyu/KRadar/calib_seq/seq_{sequence}/cam_{choose_camera}.yml'
    elif calib_seq==1:
        if sequence<10:
            path_calib = f'/home/local/xinyu/KRadar/calib_seq_v2/seq_0{sequence}/cam_{choose_camera}.yml'
        else:
            path_calib = f'/home/local/xinyu/KRadar/calib_seq_v2/seq_{sequence}/cam_{choose_camera}.yml'
    elif calib_seq==2:
        path_calib = '/home/local/xinyu/KRadar/test_maybe_right.yml'


    #flow
    for frame_idx in range(frame_idx, len(label_files), 20):
        print(f"frame_idx={frame_idx}")

        label_path= os.path.join(label_dir,label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects=info_label['objects']
        cam_front_idx = info_label['cam_front_idx']
        if choose_lidar==0:
            os1_128_idx = info_label['os1_128_idx']
        elif choose_lidar==1:
            os2_64_idx = info_label['os2_64_idx']

        boxes=torch.stack([d['box'] for d in objects],dim=0)
        if choose_lidar==0:
            pcd_path = get_lidar_path_os1(pcd_dir,os1_128_idx)
        elif choose_lidar==1:
            pcd_path = get_lidar_path_os2(pcd_dir,os2_64_idx)

        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_np = np.array(np.asarray(pcd.points))
        pcd_crop , mask= crop_lidar_points(pcd_np)
        pcd_tensor = torch.tensor(pcd_crop,dtype=torch.float32)
        
        K, distortion, R, T = load_full_camera_calib(path_calib)
        camera_points = transform_lidar_to_camera(pcd_tensor,T,R,calib_seq)

        camera_path = get_camera_path(camera_dir,cam_front_idx)
        camera_img = cv2.imread(camera_path, cv2.IMREAD_COLOR)

        img_undistort = undistort_image(
            camera_img,
            K=K,
            distortion=distortion,
            )

        lidar_corners=boxes_to_corners_3d(boxes)

        camera_corners=transform_lidar_to_camera(lidar_corners,T,R,calib_seq)

        try_with_distortion=False    
        if try_with_distortion:
            camera_2d_points,valid_mask=camera_corners_to_2d_with_distortion(camera_corners,K,distortion)
            image_with_bbx=visualize_bbx_on_camera(camera_2d_points,camera_img,valid_mask)
            #image_with_pcd=draw_pcd_on_camera(camera_points,intensity_crop,image_with_bbx, K, point_size=1)
            image_with_pcd=draw_pcd_on_camera(camera_points,image_with_bbx, K, point_size=1)  #spend a lot of time when drawing the
        else:
            camera_2d_points,valid_mask=camera_corners_to_2d_no_distortion(camera_corners,K)
            image_with_bbx=visualize_bbx_on_camera(camera_2d_points,img_undistort,valid_mask)
            # image_with_bbx=visualize_bbx_on_camera(camera_2d_points,camera_img,valid_mask)
            #image_with_pcd=draw_pcd_on_camera(camera_points,intensity_crop,image_with_bbx, K, point_size=1)
            image_with_pcd=draw_pcd_on_camera(camera_points,image_with_bbx, K, point_size=1)
            #image_with_pcd=draw_pcd_on_camera(camera_points,img_undistort, K, point_size=1)
    
        cv2.imshow("camera",image_with_pcd)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

