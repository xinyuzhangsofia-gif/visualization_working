import os
import path_setup
import numpy as np
import torch
import yaml
#os.environ.pop("QT_PLUGIN_PATH", None)
#os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
#import cv2
# import open3d as o3d
from matplotlib import pyplot as plt
from scipy.io import loadmat
from dummy_dataset import KRadarDataset
from tqdm import tqdm
import time
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


def get_4_bev_corners(radar_corners):
    radar_corners=radar_corners.cpu().numpy()
    all_unique_xyz=[]
    for corners in radar_corners:
        xy=corners[:,[0,1]]
        unique_xyz=[]
        for i,p_xy in enumerate(xy):
            is_new=True
            for q_xyz in unique_xyz:
                q_xy=q_xyz[[0,1]]
                if np.linalg.norm(p_xy-q_xy)<1e-4:
                    is_new=False
                    break
            if is_new:
                unique_xyz.append(corners[i])
        unique_xyz=np.asarray(unique_xyz,dtype=np.float32)
        all_unique_xyz.append(unique_xyz)
    all_unique_xyz = np.asarray(all_unique_xyz, dtype=np.float32)
    return all_unique_xyz


def cartesian_to_rae_advanced(all_unique_xyz):
    x = all_unique_xyz[..., 0] # x, y, z are the last dimension of lidar_corners
    y = all_unique_xyz[..., 1]
    z = all_unique_xyz[..., 2]
    r_xy = np.sqrt(x**2 + y**2) 
    r = np.sqrt(x**2 + y**2 + z**2) 
    azimuth = np.atan2(-y, x)
    azimuth = np.rad2deg(azimuth)
    elevation = np.atan2(z, r_xy)  # Add small epsilon to avoid division by zero
    rae_corners_advanced=np.stack((r, azimuth, elevation), axis=-1)
    return rae_corners_advanced


def draw_ra_bbx_2d_with_yaw(rae_corners_advanced):
    num_boxes = rae_corners_advanced.shape[0]
    bbxes_2d_advanced=[]

    for i in range(num_boxes):
        ra_points = rae_corners_advanced[i][:,[0,1]]
        bbx_2d=ra_points[:,[1,0]]
        bbx_2d = np.vstack([bbx_2d, bbx_2d[0]])
        bbxes_2d_advanced.append(bbx_2d)
    return bbxes_2d_advanced



def draw_ra_bbx_2d(rae_corners):
    
    
    num_boxes = rae_corners.shape[0]
    bbxes_2d = np.zeros((num_boxes, 5, 2), dtype=np.float32)

    for i in range(num_boxes):
        ra_points = rae_corners[i][:, [0, 1]]
        if isinstance(ra_points, torch.Tensor):
            ra_points = ra_points.detach().cpu().numpy()

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




def get_xy_extent_from_ra(arr_range, arr_azimuth_deg):
    r_min = arr_range[0]
    r_max = arr_range[-1]

    a_min = np.deg2rad(arr_azimuth_deg[0])
    a_max = np.deg2rad(arr_azimuth_deg[-1])

    angles = np.linspace(a_min, a_max, 1000)

    x_vals = r_max * np.cos(angles)
    y_vals = -r_max * np.sin(angles)

    x_vals = np.concatenate([x_vals, [0]])
    y_vals = np.concatenate([y_vals, [0]])

    x_min = np.min(x_vals)
    x_max = np.max(x_vals)
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)

    return x_min, x_max, y_min, y_max


def centers_to_edges(arr):
    arr = np.asarray(arr, dtype=np.float32)

    edges = np.zeros(len(arr) + 1, dtype=np.float32)

    edges[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    edges[0] = arr[0] - 0.5 * (arr[1] - arr[0])
    edges[-1] = arr[-1] + 0.5 * (arr[-1] - arr[-2])

    return edges


def visualize_bbx_on_ra_cartesian(
        ax,
        ra_map,
        radar_corners,
        arr_range,
        arr_azimuth_deg,
        frame_idx,
        texts
    ):
    ax.clear()

    ra_map_log = np.log10(ra_map + 1e-6)

    range_edges = centers_to_edges(arr_range)
    azimuth_edges_deg = centers_to_edges(arr_azimuth_deg)

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


    title = "RA map in Cartesian with bounding boxes"
    if frame_idx is not None:
        title += f" | frame {frame_idx}"

    ax.set_ylim(0, arr_range.max())
    ax.set_title(title)
    ax.set_xlabel("Radar x")
    ax.set_ylabel("Radar y")
    ax.set_aspect("equal")
    ax.grid(True)

    ax.set_ylim(0, arr_range.max())

    #no grid,no scale
    # ax.grid(False)
    # ax.axis("off")



def play_ra_frames_cartesian(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg,R_l2r,T_l2r,start_frame_idx):
    fig, ax = plt.subplots(figsize=(8, 6))

    #black background  
    # fig.patch.set_facecolor("black")
    # ax.set_facecolor("black") 
    for frame_idx in range(start_frame_idx,len(label_files),30):
        print(f"frame_idx = {frame_idx}")
        label_path=os.path.join(label_dir, label_files[frame_idx])

        info_label = read_info_label(label_path)
        objects = info_label['objects']
        tesseract_idx = info_label['tesseract_idx']
        texts = [f"{obj['detec_sensor']} | {obj['label']}"for obj in objects]
        ax.clear()
        if len(objects) == 0:
            ax.set_title(f"frame {frame_idx} | No objects")
            ax.text(0.5, 0.5, "No objects", ha="center", va="center", transform=ax.transAxes)
            plt.pause(0.5)
            continue
        if len(objects)>0:
            boxes=torch.stack([obj['box'] for obj in objects], dim=0)

        lidar_corners=boxes_to_corners_3d(boxes)
        radar_corners=transform_lidar_to_radar(lidar_corners,R_l2r,T_l2r)

        radar_data=radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map=radar_data['ra_map']

        visualize_bbx_on_ra_cartesian(
                                        ax,
                                        ra_map,
                                        radar_corners,
                                        arr_range,
                                        arr_azimuth_deg,
                                        frame_idx,
                                        texts
                                )
        plt.pause(0.5)


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

        


def visualize_bbx_on_ra_cartesian_out_version(
        ax,
        ra_map: np.ndarray,
        rae_corners: np.ndarray,
        arr_range: np.ndarray,
        arr_azimuth_deg: np.ndarray,
        frame_idx: int = None
    ):

    ax.clear()

    ra_map = np.log10(ra_map + 1e-6)
    azimuth_rad = np.deg2rad(arr_azimuth_deg)

    A, R = np.meshgrid(azimuth_rad, arr_range)

    ax.pcolormesh(A,R,ra_map,shading='auto',cmap='jet')
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')  # make the azimuth 0 on the right direction
    
    ax.set_thetamin(arr_azimuth_deg[0])
    ax.set_thetamax(arr_azimuth_deg[-1])
    ax.set_rlim(arr_range[0],arr_range[-1])

    angle_ticks = np.linspace(arr_azimuth_deg[0],arr_azimuth_deg[-1],9)
    ax.set_thetagrids(angle_ticks)

    range_ticks = np.linspace(arr_range[0],arr_range[-1],5)
    ax.set_rgrids(range_ticks,angle = arr_azimuth_deg[0])

    bbxes_2d = draw_ra_bbx_2d_with_yaw(rae_corners)

    for bbx_2d in bbxes_2d:
        a_deg = bbx_2d[:, 0]
        r = bbx_2d[:, 1]

        a_rad = np.deg2rad(a_deg)

        ax.plot(a_rad, r, color='r', linewidth=2)



    ax.set_title("RA polar view with bounding boxes, " + f"frame [{frame_idx}]")
    ax.grid(True,linestyle = '--',alpha = 0.4)


def play_ra_frames_cartesian_out_version(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg,R_l2r,T_l2r,start_frame_idx):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'}) # for sector

    for frame_idx in range(start_frame_idx,len(label_files)):
        print(f"frame_idx = {frame_idx}")
        label_path=os.path.join(label_dir, label_files[frame_idx])
        
        info_label = read_info_label(label_path)
        objects = info_label['objects']
        tesseract_idx = info_label['tesseract_idx']
        ax.clear()
        if len(objects) == 0:
            ax.set_title(f"frame {frame_idx} | No objects")
            ax.text(0.5, 0.5, "No objects", ha="center", va="center", transform=ax.transAxes)
            plt.pause(0.5)
            continue
        if len(objects)>0:
            boxes=torch.stack([obj['box'] for obj in objects], dim=0)

        lidar_corners=boxes_to_corners_3d(boxes)
        radar_corners=transform_lidar_to_radar(lidar_corners,R_l2r,T_l2r)
        rae_corners=cartesian_to_rae(radar_corners)

        radar_data=radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map=radar_data['ra_map']

        visualize_bbx_on_ra_cartesian_out_version(
                                ax,
                                ra_map,
                                rae_corners,
                                arr_range,
                                arr_azimuth_deg,
                                frame_idx
                            )
        plt.pause(0.5)


def play_ra_frames_polar(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg,R_l2r,T_l2r,start_frame_idx):
    fig, ax = plt.subplots(figsize=(8, 6))      #for the normal situation
    for frame_idx in range(start_frame_idx,len(label_files)):
        print(f"frame_idx = {frame_idx}")
        label_path=os.path.join(label_dir, label_files[frame_idx])

        info_label = read_info_label(label_path)
        objects = info_label['objects']
        tesseract_idx = info_label['tesseract_idx']
        texts = [f"{obj['detec_sensor']} | {obj['label']}"for obj in objects]
        ax.clear()
        if len(objects) == 0:
            ax.set_title(f"frame {frame_idx} | No objects")
            ax.text(0.5, 0.5, "No objects", ha="center", va="center", transform=ax.transAxes)
            plt.pause(0.5)
            continue
        if len(objects)>0:
            boxes=torch.stack([obj['box'] for obj in objects], dim=0)

        lidar_corners=boxes_to_corners_3d(boxes)
        radar_corners=transform_lidar_to_radar(lidar_corners,R_l2r,T_l2r)
        rae_corners=cartesian_to_rae(radar_corners)
        
        radar_data=radar_dataset.get_by_tesseract_idx(tesseract_idx)
        ra_map=radar_data['ra_map']

        visualize_bbx_on_ra_polar(ax,ra_map, rae_corners, arr_range, arr_azimuth_deg,frame_idx,texts)
        plt.pause(0.5)



def get_radar_path(radar_dataset,tesseract_idx):
    for fname in sorted(os.listdir(radar_dir)):
        if fname.startswith(f"tesseract_{tesseract_idx}"):
            return os.path.join(radar_dir,fname)
    raise FileNotFoundError(f"tesseract file not found for idx{tesseract_idx} in {radar_dataset}")


if __name__ == "__main__":
    #get and read info_label filesT
    sequence = 1
    frame_idx = 242
    choose_info_label = 'info_label_rev2' # or choose info_label_rev2
    choose_display_way = 3 # 0:one frame in polar
                           # 1:one frame in cartesian
                           # many frames in cartesian out version
                           # many frames in polar
    
    label_dir=f'/home/local/xinyu/KRadar/{sequence}/{choose_info_label}'
    label_files=sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    info_array_path = '/home/local/xinyu/KRadar/info_arr.mat'
    # time from share:2.7s time from xinyu:1.7s
    radar_dataset=KRadarDataset("/home/local/xinyu/KRadar/1/radar_tesseract")
    radar_dataset=KRadarDataset("/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar/1/radar_tesseract")
    radar_dir = "/home/local/xinyu/KRadar/1/radar_tesseract"
    radar_dir = "/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar/1/radar_tesseract"
    arr_range,arr_azimuth_deg, arr_elevation_deg =load_axis_from_mat(info_array_path)
    lidar2radar_calib_path = "/home/local/xinyu/MVRSS/mvrss/lidar2radar_calib.yml"
    R_l2r,T_l2r = load_lidar2radar_calib(lidar2radar_calib_path)



    # only show one frame
    label_path = os.path.join(label_dir, label_files[frame_idx])
    info_label = read_info_label(label_path)

    objects = info_label["objects"]
    tesseract_idx = info_label["tesseract_idx"]
    texts = [f"{obj['detec_sensor']} | {obj['label']}"for obj in objects]
    
    radar_data = radar_dataset.get_by_tesseract_idx(tesseract_idx)
    ra_map = radar_data["ra_map"]
    
    boxes = torch.stack([obj["box"] for obj in objects], dim=0)

    lidar_corners = boxes_to_corners_3d(boxes)

    radar_corners = transform_lidar_to_radar(
        lidar_corners,
        R_l2r,
        T_l2r
    )

    rae_corners = cartesian_to_rae(radar_corners)
    

    #new choice
    # all_unique_xyz=get_4_bev_corners(radar_corners)
    # rae_corners = cartesian_to_rae_advanced(all_unique_xyz)

    start_frame_idx=0

    if choose_display_way == 0:

        fig, ax = plt.subplots(figsize=(8, 6))

        visualize_bbx_on_ra_polar(
            ax=ax,
            ra_map=ra_map,
            rae_corners=rae_corners,
            arr_range=arr_range,
            arr_azimuth_deg=arr_azimuth_deg,
            frame_idx=frame_idx,
            texts=texts
        )
        plt.show()

    elif choose_display_way==2:
        fig, ax = plt.subplots(figsize=(8, 6))

        visualize_bbx_on_ra_cartesian_out_version(
            ax=ax,
            ra_map=ra_map,
            rae_corners=rae_corners,
            arr_range=arr_range,
            arr_azimuth_deg=arr_azimuth_deg,
            frame_idx=frame_idx
        )
        plt.show()
    elif choose_display_way==1:
        fig, ax = plt.subplots(figsize=(8, 6))
        visualize_bbx_on_ra_cartesian(
                                                ax,
                                                ra_map=ra_map,
                                                radar_corners=radar_corners,
                                                arr_range=arr_range,
                                                arr_azimuth_deg=arr_azimuth_deg,
                                                frame_idx=frame_idx,
                                                texts=texts
                                            )
        plt.show()
    elif choose_display_way==3:
        play_ra_frames_polar(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg,R_l2r,T_l2r,start_frame_idx)
    elif choose_display_way==4:
        play_ra_frames_cartesian(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg,R_l2r,T_l2r,start_frame_idx)
    elif choose_display_way==5:
        play_ra_frames_cartesian_out_version(label_dir,label_files,radar_dataset,arr_range,arr_azimuth_deg,R_l2r,T_l2r,start_frame_idx)


 


    

   
    
    
