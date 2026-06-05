import os
import path_setup
import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
#os.environ.pop("QT_PLUGIN_PATH", None)
#os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
import cv2
from scipy.io import loadmat
from dummy_dataset import KRadarDataset
from tqdm import tqdm

def read_info_label(label_path):
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

        #read the first row and get the index relationship betweeen radar,camera,lidar
        header = lines[0]
        idx_str = header.split('=')[-2]
        idx_part = idx_str.split('_')

        tesseract_idx = idx_part[0]
        os2_64_idx = idx_part[1]
        cam_front_idx = idx_part[2]
        os1_128_idx = idx_part[3]

        #read the data about the bbx
        objects = []
        for line in lines[1:]:  # Skip the first line (header)
            parts = [p.strip() for p in line.split(',')]
            
            detec_sensor=parts[1]
            label=parts[2]
            cls=parts[3]
            x=float(parts[4])
            y=float(parts[5])
            z=float(parts[6])
            yaw=float(parts[7])*np.pi/180.0  # Convert yaw from degrees to radians
            l=2*float(parts[8])
            w=2*float(parts[9])
            h=2*float(parts[10])

            box=torch.tensor([x, y, z, l, w, h, yaw],dtype=torch.float32)

            objects.append({
                'detec_sensor':detec_sensor,
                'label':label,
                'cls': cls,
                'box': box
            })        
    return {
        'objects': objects,
        'tesseract_idx':tesseract_idx,
        'os2_64_idx':os2_64_idx,
        'cam_front_idx':cam_front_idx,
        'os1_128_idx': os1_128_idx
    }


def get_radar_path(radar_dataset,tesseract_idx):
    for fname in sorted(os.listdir(radar_dir)):
        if fname.startswith(f"tesseract_{tesseract_idx}"):
            return os.path.join(radar_dir,fname)
    raise FileNotFoundError(f"tesseract file not found for idx{tesseract_idx} in {radar_dataset}")


def boxes_to_corners_3d(boxes):#from kradar,box_utils.py
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


def draw_ra_bbx_2d(rae_corners):
    
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


def visualize_bbx_on_ra_polar(ax,
                        ra_map, 
                        rae_corners,
                        arr_range, 
                        arr_azimuth_deg,
                        frame_idx,
                        texts):
    ax.clear()
    
    ra_map = np.log10(ra_map + 1e-6)  # Add small epsilon to avoid log(0)

    ax.imshow(ra_map,
               origin='lower', 
               aspect='auto', 
               cmap='jet',
               extent=[arr_azimuth_deg[0], arr_azimuth_deg[-1], arr_range[0], arr_range[-1]])


    bbxes_2d = draw_ra_bbx_2d(rae_corners)
    for box_idx,bbx_2d in enumerate(bbxes_2d):
        ax.plot(
            bbx_2d[:, 0],
            bbx_2d[:, 1],
            color="r",
            linewidth=2
        )
        if texts is not None and box_idx < len(texts):
            text = texts[box_idx]
        else:
            text = f"obj {box_idx}"

        text_x = bbx_2d[:-1, 0].mean()
        text_y = bbx_2d[:-1, 1].max()

        ax.text(
            text_x,
            text_y + 0.8,
            text,
            color="red",
            fontsize=9,
            ha="center",
            va="bottom",
            )


    title = "RA map with bounding boxes"
    if frame_idx is not None:
        title += f" | frame {frame_idx}"

    ax.set_title(title)
    ax.set_ylabel("Range")
    ax.set_xlabel("Azimuth")
    ax.set_xlim(arr_azimuth_deg[0], arr_azimuth_deg[-1])
    ax.set_ylim(arr_range[0], arr_range[-1])
    ax.grid(True)


def preload_ra_polar_frames(    
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        max_frames,
        R_l2r,
        T_l2r,
        start_frame_idx=0
    ):

    frames = []
    
  
    for frame_idx in tqdm(range(start_frame_idx,max_frames), desc="Preloading frames"):

        label_path= os.path.join(label_dir,label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects=info_label['objects']
        tesseract_idx = info_label["tesseract_idx"]
        texts = [
            f"{obj['detec_sensor']} | {obj['label']}"
            for obj in objects
        ]

        radar_data = radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map = radar_data["ra_map"]
        fig, ax = plt.subplots(figsize=(8, 6))
        if len(objects) > 0:
            boxes = torch.stack([obj["box"] for obj in objects], dim=0)
            lidar_corners = boxes_to_corners_3d(boxes)
            radar_corners = transform_lidar_to_radar(
                lidar_corners,
                R_l2r,
                T_l2r
            )
            rae_corners = cartesian_to_rae(radar_corners)

        else:
            rae_corners = np.zeros((0, 8, 3), dtype=np.float32)
            texts=[]
        
        visualize_bbx_on_ra_polar(
            ax,
            ra_map,
            rae_corners,
            arr_range,
            arr_azimuth_deg,
            frame_idx,
            texts=texts
        )

        image = fig_to_cv2_image(fig)
        plt.close(fig)
        frames.append(image)

    print(f"Preload finished. Total frames: {len(frames)}")

    return frames


def play_ra_polar_frames(
        frames,
        fps=10,
        window_name="RA map with bounding boxes",
        save_path=None
    ):

    delay = int(1000 / fps)

    h, w = frames[0].shape[:2]
     
    writer = None
    if save_path is not None and save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {save_path}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    num_frames = len(frames)

    while frame_idx < num_frames:

        image = frames[frame_idx]

        if writer is not None:
            writer.write(image)

        cv2.imshow(window_name, image)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:  #27 means ESC
            break

        if key == ord(" "):
            print("Paused. Press SPACE to continue, q/ESC to quit.")

            while True:
                key2 = cv2.waitKey(0) & 0xFF

                if key2 == ord(" "):
                    break

                if key2 == ord("q") or key2 == 27:
                    cv2.destroyAllWindows()
                    return

        frame_idx += 1
    
    if writer is not None:
        writer.release()

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)



def play_ra_cartesian_frames(
        frames,
        fps=10,
        window_name="RA map in cartesian with bounding boxes",
        save_path=None
    ):

    delay = int(1000 / fps)
    
    h, w = frames[0].shape[:2]
    
    writer = None
    if save_path is not None and save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {save_path}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    num_frames = len(frames)

    while frame_idx < num_frames:

        image = frames[frame_idx]
        cv2.imshow(window_name, image)
        if writer is not None:
            writer.write(image)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:  #27 means ESC
            break

        if key == ord(" "):
            print("Paused. Press SPACE to continue, q/ESC to quit.")

            while True:
                key2 = cv2.waitKey(0) & 0xFF

                if key2 == ord(" "):
                    break

                if key2 == ord("q") or key2 == 27:
                    cv2.destroyAllWindows()
                    return

        frame_idx += 1

    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def get_ra_cartesian_limits(arr_range, arr_azimuth_deg):
    r_max = arr_range.max()

    a_min = np.deg2rad(arr_azimuth_deg.min())
    a_max = np.deg2rad(arr_azimuth_deg.max())

    x_min = r_max * np.sin(a_min)
    x_max = r_max * np.sin(a_max)

    y_min = 0
    y_max = r_max * np.cos(0)

    return x_min, x_max, y_min, y_max

def visualize_bbx_on_ra_cartesian(
        ax,
        ra_map: np.ndarray,
        radar_corners,
        arr_range: np.ndarray,
        arr_azimuth_deg: np.ndarray,
        frame_idx: int = None,
        texts=None
    ):
    ax.clear()

    ra_map_log = np.log10(ra_map + 1e-6)

    # center to edges
    range_edges = np.zeros(len(arr_range) + 1, dtype=np.float32)
    range_edges[1:-1] = 0.5 * (arr_range[:-1] + arr_range[1:])
    range_edges[0] = arr_range[0] - 0.5 * (arr_range[1] - arr_range[0])
    range_edges[-1] = arr_range[-1] + 0.5 * (arr_range[-1] - arr_range[-2])

    azimuth_edges_deg = np.zeros(len(arr_azimuth_deg) + 1, dtype=np.float32)
    azimuth_edges_deg[1:-1] = 0.5 * (arr_azimuth_deg[:-1] + arr_azimuth_deg[1:])
    azimuth_edges_deg[0] = arr_azimuth_deg[0] - 0.5 * (arr_azimuth_deg[1] - arr_azimuth_deg[0])
    azimuth_edges_deg[-1] = arr_azimuth_deg[-1] + 0.5 * (arr_azimuth_deg[-1] - arr_azimuth_deg[-2])
    R_edge, A_edge = np.meshgrid(
        range_edges,
        np.deg2rad(azimuth_edges_deg),
        indexing="ij"
    )

    X_edge = R_edge * np.sin(A_edge)
    Y_edge = R_edge * np.cos(A_edge)

    ax.pcolormesh(
        X_edge,
        Y_edge,
        ra_map_log,
        shading="flat",
        cmap="jet"
    )

    for box_idx, corners in enumerate(radar_corners):

        
        x3d = corners[:, 0]
        y3d = corners[:, 1]

        pts_2d = torch.stack([
            -y3d,
            x3d
        ], dim=1)   # shape = (8, 2)


        pts_np = pts_2d.detach().cpu().numpy()

        x_min = np.min(pts_np[:, 0])
        x_max = np.max(pts_np[:, 0])
        y_min = np.min(pts_np[:, 1])
        y_max = np.max(pts_np[:, 1])

        bbx_2d = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min]
        ], dtype=np.float32)

        ax.plot(
            bbx_2d[:, 0],
            bbx_2d[:, 1],
            color="r",
            linewidth=2
        )
        
        if texts is not None and box_idx < len(texts):
            text = texts[box_idx]
        else:
            text = f"obj {box_idx}"

        text_x = bbx_2d[:-1, 0].mean()
        text_y = bbx_2d[:-1, 1].max()

        ax.text(
            text_x,
            text_y + 0.8,
            text,
            color="red",
            fontsize=9,
            ha="center",
            va="bottom",
            )


    # title = "RA map in Cartesian with bounding boxes"
    # if frame_idx is not None:
    #     title += f" | frame {frame_idx}"

    x_min, x_max, y_min, y_max = get_ra_cartesian_limits(arr_range,arr_azimuth_deg)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max+10)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Radar y")
    ax.set_ylabel("Radar x")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")
    #ax.set_title(title,color='white')


def visualize_bbx_on_ra_cartesian_with_yaw(
        ax,
        ra_map: np.ndarray,
        radar_corners,
        arr_range: np.ndarray,
        arr_azimuth_deg: np.ndarray,
        frame_idx: int = None,
        texts=None
    ):
    ax.clear()

    ra_map_log = np.log10(ra_map + 1e-6)

    # range_edges = centers_to_edges(arr_range)
    # azimuth_edges_deg = centers_to_edges(arr_azimuth_deg)

    # center to edges
    range_edges = np.zeros(len(arr_range) + 1, dtype=np.float32)
    range_edges[1:-1] = 0.5 * (arr_range[:-1] + arr_range[1:])
    range_edges[0] = arr_range[0] - 0.5 * (arr_range[1] - arr_range[0])
    range_edges[-1] = arr_range[-1] + 0.5 * (arr_range[-1] - arr_range[-2])

    azimuth_edges_deg = np.zeros(len(arr_azimuth_deg) + 1, dtype=np.float32)
    azimuth_edges_deg[1:-1] = 0.5 * (arr_azimuth_deg[:-1] + arr_azimuth_deg[1:])
    azimuth_edges_deg[0] = arr_azimuth_deg[0] - 0.5 * (arr_azimuth_deg[1] - arr_azimuth_deg[0])
    azimuth_edges_deg[-1] = arr_azimuth_deg[-1] + 0.5 * (arr_azimuth_deg[-1] - arr_azimuth_deg[-2])
    R_edge, A_edge = np.meshgrid(
        range_edges,
        np.deg2rad(azimuth_edges_deg),
        indexing="ij"
    )

    X_edge = R_edge * np.sin(A_edge)
    Y_edge = R_edge * np.cos(A_edge)

    ax.pcolormesh(
        X_edge,
        Y_edge,
        ra_map_log,
        shading="flat",
        cmap="jet"
    )

    for box_idx, corners in enumerate(radar_corners):

        
        x3d = corners[:, 0]
        y3d = corners[:, 1]

        pts_2d = torch.stack([
            -y3d,
            x3d
        ], dim=1)   # shape = (8, 2)


        pts_np = pts_2d.detach().cpu().numpy()

        unique_pts = []

        tol = 1e-4

        for p in pts_np:
            is_new = True
            for q in unique_pts:
                if np.linalg.norm(p - q) < tol:
                    is_new = False
                    break

            if is_new:
                unique_pts.append(p)

        unique_pts = np.asarray(unique_pts, dtype=np.float32)

        if unique_pts.shape[0] != 4:
            print(f"Warning: expected 4 unique BEV corners, got {unique_pts.shape[0]}")
            continue

        center = unique_pts.mean(axis=0)

        angles = np.arctan2(
            unique_pts[:, 1] - center[1],
            unique_pts[:, 0] - center[0]
        )

        order = np.argsort(angles)
        bbx_2d = unique_pts[order]


        bbx_2d = np.vstack([bbx_2d, bbx_2d[0]])

        ax.plot(
            bbx_2d[:, 0],
            bbx_2d[:, 1],
            color="r",
            linewidth=2
        )
        
        if texts is not None and box_idx < len(texts):
            text = texts[box_idx]
        else:
            text = f"obj {box_idx}"

        text_x = bbx_2d[:-1, 0].mean()
        text_y = bbx_2d[:-1, 1].max()

        ax.text(
            text_x,
            text_y + 0.8,
            text,
            color="red",
            fontsize=9,
            ha="center",
            va="bottom",
            )


    # title = "RA map in Cartesian with bounding boxes"
    # if frame_idx is not None:
    #     title += f" | frame {frame_idx}"

    x_min, x_max, y_min, y_max = get_ra_cartesian_limits(arr_range,arr_azimuth_deg)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max+10)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Radar y")
    ax.set_ylabel("Radar x")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")
    #ax.set_title(title,color='white')



def fig_to_cv2_image(fig):
    fig.canvas.draw()

    img_rgb = np.asarray(fig.canvas.buffer_rgba())
    img_rgb = img_rgb[:, :, :3]

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    return img_bgr

def preload_ra_cartesian_frames(  
                                    label_dir,
                                    label_files,
                                    radar_dataset,
                                    arr_range,
                                    arr_azimuth_deg,
                                    max_frames,
                                    R_l2r,
                                    T_l2r,
                                    start_frame_idx=0
                                ):

    frames = []
    
    for frame_idx in tqdm(range(start_frame_idx,max_frames), desc="Preloading frames"):

        label_path= os.path.join(label_dir,label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects=info_label['objects']
        tesseract_idx = info_label["tesseract_idx"]
        texts = [
            f"{obj['detec_sensor']} | {obj['label']}"
            for obj in objects
        ]

        radar_data = radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map = radar_data["ra_map"]

        if len(objects) > 0:
            boxes = torch.stack([obj["box"] for obj in objects], dim=0)
            lidar_corners = boxes_to_corners_3d(boxes)

            radar_corners = transform_lidar_to_radar(
                lidar_corners,
                R_l2r,
                T_l2r
            )

        else:
            radar_corners = torch.zeros((0, 8, 3), dtype=torch.float32)

        # fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        x_min, x_max, y_min, y_max = get_ra_cartesian_limits(
            arr_range,
            arr_azimuth_deg
        )

        data_w = x_max - x_min
        data_h = y_max - y_min + 5

        fig_w = 8
        fig_h = fig_w * data_h / data_w

        title = "RA map in Cartesian with bounding boxes"
        title += f" | frame {frame_idx}"

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=120, facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1], facecolor="black")

        fig.text(0.5,0.96,title,color="white",ha="center",va="center",fontsize=12)

        #black background
        # fig.patch.set_facecolor("black")
        # ax.set_facecolor("black")
        visualize_bbx_on_ra_cartesian(
                ax,
                ra_map,
                radar_corners,
                arr_range,
                arr_azimuth_deg,
                frame_idx,
                texts=texts
            )
        image = fig_to_cv2_image(fig)
        plt.close(fig)
        frames.append(image)

    print(f"Preload finished. Total frames: {len(frames)}")

    return frames

def preload_ra_cartesian_frames_with_yaw(  
                                    label_dir,
                                    label_files,
                                    radar_dataset,
                                    arr_range,
                                    arr_azimuth_deg,
                                    max_frames,
                                    R_l2r,
                                    T_l2r,
                                    start_frame_idx=0
                                ):

    frames = []
    
    for frame_idx in tqdm(range(start_frame_idx,max_frames), desc="Preloading frames"):

        label_path= os.path.join(label_dir,label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects=info_label['objects']
        tesseract_idx = info_label["tesseract_idx"]
        texts = [
            f"{obj['detec_sensor']} | {obj['label']}"
            for obj in objects
        ]

        radar_data = radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map = radar_data["ra_map"]

        if len(objects) > 0:
            boxes = torch.stack([obj["box"] for obj in objects], dim=0)
            lidar_corners = boxes_to_corners_3d(boxes)

            radar_corners = transform_lidar_to_radar(
                lidar_corners,
                R_l2r,
                T_l2r
            )

        else:
            radar_corners = torch.zeros((0, 8, 3), dtype=torch.float32)

        # fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        x_min, x_max, y_min, y_max = get_ra_cartesian_limits(
            arr_range,
            arr_azimuth_deg
        )

        data_w = x_max - x_min
        data_h = y_max - y_min + 5

        fig_w = 8
        fig_h = fig_w * data_h / data_w

        title = "RA map in Cartesian with bounding boxes"
        title += f" | frame {frame_idx}"

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=120, facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1], facecolor="black")

        fig.text(0.5,0.96,title,color="white",ha="center",va="center",fontsize=12)

        #black background
        # fig.patch.set_facecolor("black")
        # ax.set_facecolor("black")

        visualize_bbx_on_ra_cartesian(
            ax,
            ra_map,
            radar_corners,
            arr_range,
            arr_azimuth_deg,
            frame_idx,
            texts=texts
        )
        image = fig_to_cv2_image(fig)
        plt.close(fig)
        frames.append(image)

    print(f"Preload finished. Total frames: {len(frames)}")

    return frames


if __name__ == "__main__":

    sequence = 1
    frame_idx = 0
    choose_info_label = 'info_label_rev2' # or choose info_label_rev2
    display_form = 0     #0:polar  1:cartesian  2:cartesian_with_yaw
    fps=10
    start_frame_idx=0   #use in video

    label_dir=f'/home/local/xinyu/KRadar/{sequence}/{choose_info_label}'
    label_files=sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    info_array_path = f'/home/local/xinyu/KRadar/info_arr.mat'
    lidar2radar_calib_path = "/home/local/xinyu/MVRSS/mvrss/lidar2radar_calib.yml"
    
    max_frames = len(label_files)
   # max_frames = 552
    
    radar_dataset=KRadarDataset(f"/home/local/xinyu/KRadar/{sequence}/radar_tesseract")
    radar_dir = f"/home/local/xinyu/KRadar/{sequence}/radar_tesseract"

    radar_dataset=KRadarDataset(
        f"/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar/{sequence}/radar_tesseract")
    radar_dir = f"/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar/{sequence}/radar_tesseract"

    arr_range,arr_azimuth_deg, arr_elevation_deg =load_axis_from_mat(info_array_path)
    R_l2r,T_l2r = load_lidar2radar_calib(lidar2radar_calib_path)
    
    if display_form==0:
        frames = preload_ra_polar_frames(
            label_dir,
            label_files,
            radar_dataset,
            arr_range,
            arr_azimuth_deg,
            max_frames,
            R_l2r,
            T_l2r,
            start_frame_idx
        )

        play_ra_polar_frames(
            frames=frames,
            fps=fps,
            save_path = 'ra_polar_video.mp4' 
        )
    
    elif display_form==1:
            frames = preload_ra_cartesian_frames(
                label_dir,
                label_files,
                radar_dataset,
                arr_range,
                arr_azimuth_deg,
                max_frames,
                R_l2r,
                T_l2r,
                start_frame_idx
            )

            play_ra_cartesian_frames(
                frames=frames,
                fps=fps,
                save_path = ''
           )
    elif display_form==2:
            frames = preload_ra_cartesian_frames_with_yaw(
                label_dir,
                label_files,
                radar_dataset,
                arr_range,
                arr_azimuth_deg,
                max_frames,
                R_l2r,
                T_l2r,
                start_frame_idx
            )

            play_ra_cartesian_frames(
                frames=frames,
                fps=fps,
                save_path = ''
           )

    

   
    
    
