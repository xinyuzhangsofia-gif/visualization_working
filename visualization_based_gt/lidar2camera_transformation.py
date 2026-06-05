import os
import path_setup

os.environ.pop("QT_QPA_PLATFORM", None)
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ["MPLBACKEND"] = "TkAgg"

import matplotlib
matplotlib.use("TkAgg")

from lidar2radar_transformation_video import read_info_label,boxes_to_corners_3d
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
import os
import cv2
import numpy as np
import yaml


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


def add_label_to_camera_bbx(
        image,
        text_x,
        text_y,
        text,
        font_size=0.5,
        y_offset=10
    ):

    if text is None:
        return

    text_x = int(text_x)
    text_y = int(text_y)

    # Put text above the box
    text_y = max(20, text_y - y_offset)

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 0),   # green, BGR
        1,
        cv2.LINE_AA
    )

def visualize_bbx_on_camera(camera_2d_points, image,valid_mask,texts=None):

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
            if not (box_valid[start] and box_valid[end]):
                continue

            point1 = box[start]
            point2 = box[end]

            if not (np.isfinite(point1).all() and np.isfinite(point2).all()):
                continue

            if not (-1000 < point1[0] < w + 1000 and -1000 < point1[1] < h + 1000):
                continue
            if not (-1000 < point2[0] < w + 1000 and -1000 < point2[1] < h + 1000):
                continue

            x1, y1 = np.round(point1).astype(np.int32)
            x2, y2 = np.round(point2).astype(np.int32)

            cv2.line(image_with_bbx, (x1, y1), (x2, y2), (0, 255, 0), 1,cv2.LINE_AA)

        if texts is not None and i < len(texts):
            valid_points = box[box_valid]

        if valid_points.shape[0] > 0:
            valid_points = valid_points[np.isfinite(valid_points).all(axis=1)]

            if valid_points.shape[0] > 0:
                x_min = np.min(valid_points[:, 0])
                y_min = np.min(valid_points[:, 1])

                # Clip text position inside image
                text_x = int(np.clip(x_min, 0, w - 1))
                text_y = int(np.clip(y_min, 0, h - 1))

                add_label_to_camera_bbx(
                    image_with_bbx,
                    text_x,
                    text_y,
                    texts[i],
                    font_size=0.3,
                    y_offset=10
                )


    return image_with_bbx

def play_camera_video(
        frame_idx=0,
        sequence=1,
        choose_info_label='info_label_rev2',
        choose_camera='cam_1',
        calib_seq='calib_seq_v2',
        step=1,
        fps=10
    ):

    delay = int(1000 / fps)

    label_dir = f'/home/local/xinyu/KRadar/{sequence}/{choose_info_label}'
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    camera_dir = f'/home/local/xinyu/KRadar/{sequence}/{choose_camera}_front'
   
    if sequence < 10:
        path_calib = f'/home/local/xinyu/KRadar/{calib_seq}/seq_0{sequence}/{choose_camera}.yml'
    else:
        path_calib = f'/home/local/xinyu/KRadar/{calib_seq}/seq_{sequence}/{choose_camera}.yml'
    #path_calib = '/home/local/xinyu/KRadar/test_maybe_right.yml'
   
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    for frame_idx in range(frame_idx, len(label_files), step):
        print(f"frame_idx = {frame_idx}")

        label_path = os.path.join(label_dir, label_files[frame_idx])
        info_label = read_info_label(label_path)

        objects = info_label['objects']
        cam_front_idx = info_label['cam_front_idx']
        texts = [f"{obj['detec_sensor']} | {obj['label']}"for obj in objects]
        
        K, distortion, R, T = load_full_camera_calib(path_calib)
        
        camera_path = get_camera_path(camera_dir, cam_front_idx)
        camera_img = cv2.imread(camera_path, cv2.IMREAD_COLOR)
        
        boxes = torch.stack([d['box'] for d in objects], dim=0)
        
        lidar_corners = boxes_to_corners_3d(boxes)
        camera_corners = transform_lidar_to_camera(lidar_corners,T,R)

        img_undistort = undistort_image(
            camera_img,
            K=K,
            distortion=distortion,
            )
        
        camera_2d_points,valid_mask=camera_corners_to_2d_undistort(camera_corners,K)
        image_with_bbx=visualize_bbx_on_camera(camera_2d_points,img_undistort,valid_mask,texts)

        cv2.imshow("camera", image_with_bbx)
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            print("Paused. Press SPACE to continue, q to quit.")
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord(" "):
                    break
                if key2 == ord("q"):
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()


if __name__ == "__main__":

    #data_choose
    frame_idx = 0
    sequence=1
    choose_info_label= 'info_label_rev2'  #info_label , info_label_rev2
    choose_camera='cam_1'      # cam_1, cam_2
    choose_lidar='os2-64'       #0:os1-128 ,1:os2-64
    calib_seq = 'calib_seq_v2'     #calib_seq , calib_seq_v2
    display_mode = 1              #0:visualize with bbx
                                  #1:video with bbx
    label_dir = f'/home/local/xinyu/KRadar/{sequence}/{choose_info_label}'
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    camera_dir = f'/home/local/xinyu/KRadar/{sequence}/{choose_camera}_front'
    if sequence < 10:
        path_calib = f'/home/local/xinyu/KRadar/{calib_seq}/seq_0{sequence}/{choose_camera}.yml'
    else:
        path_calib = f'/home/local/xinyu/KRadar/{calib_seq}/seq_{sequence}/{choose_camera}.yml'
    if display_mode ==0:
        for frame_idx in range(frame_idx, len(label_files), 20):
            print(f"frame_idx={frame_idx}")
        
            label_path = os.path.join(label_dir, label_files[frame_idx])
            info_label = read_info_label(label_path)

            objects = info_label['objects']
            cam_front_idx = info_label['cam_front_idx']
            texts = [f"{obj['detec_sensor']} | {obj['label']}"for obj in objects]
            
            K, distortion, R, T = load_full_camera_calib(path_calib)
            
            camera_path = get_camera_path(camera_dir, cam_front_idx)
            camera_img = cv2.imread(camera_path, cv2.IMREAD_COLOR)
            
            boxes = torch.stack([d['box'] for d in objects], dim=0)
            
            lidar_corners = boxes_to_corners_3d(boxes)
            camera_corners = transform_lidar_to_camera(lidar_corners,T,R)

            img_undistort = undistort_image(
                camera_img,
                K=K,
                distortion=distortion,
                )
            
            camera_2d_points,valid_mask=camera_corners_to_2d_undistort(camera_corners,K)
            image_with_bbx=visualize_bbx_on_camera(camera_2d_points,img_undistort,valid_mask,texts)

            cv2.imshow("camera",image_with_bbx)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif display_mode==1:
        play_camera_video(
                        frame_idx,
                        sequence,
                        choose_info_label,
                        choose_camera,
                        calib_seq,
                        step=1,
                        fps=10
                    )
