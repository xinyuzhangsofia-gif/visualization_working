import os
import path_setup
import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import time
import open3d as o3d
from matplotlib import pyplot as plt
from tqdm import tqdm
from zxy_label_utils import *
from zxy_data_path import *
from sensor_transformation import *
from PIL import Image, ImageDraw, ImageFont


# Lidar Visualization PointCloud and BeV

def draw_bbx_lines(lidar_corners):
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
        ]

    line_sets=[]

    for corners in lidar_corners:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)

        colors=[[1,0,0]for _ in edges]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        line_sets.append(line_set)

    return line_sets


def create_text_mesh(
        text,
        position,
        scale=0.015,
        color=(1.0, 1.0, 1.0),
        font_size=18,
        bold_offset=0,
        sample_step=1,
        rotate_deg=-90
    ):

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    font = ImageFont.truetype(
        font_path,
        size=font_size
    )

    tmp_img = Image.new("L", (1, 1), 0)
    tmp_draw = ImageDraw.Draw(tmp_img)

    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img_w = text_w + 20
    img_h = text_h + 20

    img = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(img)

    for dx in range(-bold_offset, bold_offset + 1):
        for dy in range(-bold_offset, bold_offset + 1):
            draw.text(
                (10 + dx, 10 + dy),
                text,
                fill=255,
                font=font
            )

    img_np = np.asarray(img)

    vertices = []
    triangles = []

    theta = np.deg2rad(rotate_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    ys, xs = np.where(img_np > 0)

    for x, y in zip(xs[::sample_step], ys[::sample_step]):
        local_x = (x - img_w / 2) * scale
        local_y = -(y - img_h / 2) * scale

        half = scale * 0.55

        local_corners = [
            [local_x - half, local_y - half],
            [local_x + half, local_y - half],
            [local_x + half, local_y + half],
            [local_x - half, local_y + half]
        ]

        rotated_vertices = []

        for lx, ly in local_corners:
            rx = lx * cos_t - ly * sin_t
            ry = lx * sin_t + ly * cos_t

            rotated_vertices.append([
                position[0] + rx,
                position[1] + ry,
                position[2]
            ])

        base_idx = len(vertices)

        vertices.extend(rotated_vertices)

        triangles.append([base_idx, base_idx + 1, base_idx + 2])
        triangles.append([base_idx, base_idx + 2, base_idx + 3])

    mesh = o3d.geometry.TriangleMesh()

    if len(vertices) == 0:
        return mesh

    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=np.float64)
    )

    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(triangles, dtype=np.int32)
    )

    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh


def create_bbx_text_geometries(
        lidar_corners,
        texts=None,
        z_offset=0.3,
        outside_offset=0.5,
        text_scale=0.015,
        text_color=(1.0, 1.0, 1.0),
        rotate_deg=-90
    ):

    lidar_corners = lidar_corners.cpu().numpy()

    num_boxes = lidar_corners.shape[0]

    if texts is None:
        texts = [f"obj_{i}" for i in range(num_boxes)]

    text_geometries = []

    for i, corners in enumerate(lidar_corners):
        x_max = corners[:, 0].max()
        y_center = corners[:, 1].mean()
        z_max = corners[:, 2].max()

        text_position = np.array([
            x_max + outside_offset,
            y_center,
            z_max + z_offset
        ])

        text_mesh = create_text_mesh(
            text=texts[i],
            position=text_position,
            scale=text_scale,
            color=text_color,
            font_size=18,
            bold_offset=0,
            sample_step=1,
            rotate_deg=rotate_deg
        )

        text_geometries.append(text_mesh)

    return text_geometries


def visualize_bbx_on_lidar_pcd(lidar_corners,pcd,texts,show_texts):
    geometries = []
    
    axis=o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    geometries.append(axis)
    geometries.append(pcd)

    bbox_lines = draw_bbx_lines(lidar_corners)
    geometries.extend(bbox_lines)
    

    if show_texts:
        text_geometries = create_bbx_text_geometries(
            lidar_corners,
            texts=texts,
            z_offset=0.3,
            outside_offset=1.0,
            text_scale=0.1,
            text_color=(1.0, 0.0, 0.0),
            rotate_deg=-90
        )
        geometries.extend(text_geometries)


    all_points = (lidar_corners.cpu().numpy()).reshape(-1,3)
    center = all_points.mean(axis=0)
    
    def view_xy(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([0,0,-1])
        view_control.set_up([0,1,0])
        view_control.set_zoom(1)
        return False         
    
    def view_xz(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([0,-1,0])
        view_control.set_up([0,0,1])
        view_control.set_zoom(1)
        return False
    
    def view_yz(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([-1,0,0])
        view_control.set_up([0,0,1])
        view_control.set_zoom(1)
        return False
    
    def view_special(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([-1,0.05,0.25])
        view_control.set_up([0,0,1])
        view_control.set_zoom(1)
        return False
    
    def view_bev(vision):
        view_control = vision.get_view_control()
        view_control.set_lookat(center)
        view_control.set_front([0, 0, 1])
        view_control.set_up([1, 0, 0])
        view_control.set_zoom(1)
        return False
    
    key_to_callback = {
        ord("1"):view_xy,
        ord("2"):view_xz,
        ord("3"):view_yz,
        ord("4"):view_special,
        ord('5'):view_bev
    }

                             
    o3d.visualization.draw_geometries_with_key_callbacks(geometries,
                                                         key_to_callback,
                                                         window_name = "Press 1:XY 2:XZ 3:YZ 4:suitable view 5:bev" )


def show_single_lidar_pcd(
        label_dir,
        lidar_dir,
        lidar_type,
        frame_idx=0,
        show_texts=False
    ):

    label_files = get_label_files(label_dir)

    for frame_idx in range(frame_idx, len(label_files)):

        print(f"frame_idx = {frame_idx}")

        label_path = os.path.join(label_dir, label_files[frame_idx])

        info_label = read_info_label(label_path)
        objects = info_label["objects"]
        texts = [f"{obj['detec_sensor']} | {obj['label']}"for obj in objects]

        lidar_idx = get_lidar_idx(info_label,lidar_type)
        lidar_path = get_lidar_path(lidar_dir,lidar_type,lidar_idx)

        pcd = o3d.io.read_point_cloud(lidar_path)
        if len(objects) == 0:
            print(f"frame_idx = {frame_idx}, no objects")
            continue
        boxes = torch.stack([obj["box"] for obj in objects],dim=0)
        lidar_corners = boxes_to_corners_3d(boxes)
       
        visualize_bbx_on_lidar_pcd(
            lidar_corners,
            pcd,
            texts,
            show_texts
        )


def play_bev_lidar_video(
        label_dir,
        lidar_dir,
        lidar_type,
        start_frame_idx=0,
        fps=10,
        show_texts=False
    ):
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

    paused = False
    should_quit = False

    def pause_callback(vision):
        nonlocal paused
        paused = not paused
        return False

    def quit_callback(vision):
        nonlocal should_quit
        should_quit = True
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="BEV LiDAR Video",
        width=720,
        height=720
    )
    vis.register_key_callback(ord(" "), pause_callback)
    vis.register_key_callback(ord("Q"), quit_callback)
    vis.register_key_callback(256, quit_callback)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    old_geometries = []

    frame_interval = 1.0 / fps

    # fixed BEV center
    fixed_center = np.array([20.0, 0.0, 0.0])
    need_reset = True
    frame_idx = start_frame_idx

    while frame_idx < len(label_files):
        if should_quit:
            break

        while paused and not should_quit:
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)

        if should_quit:
            break

        if frame_idx%50 == 0:
            print(f"Playing frame_idx = {frame_idx}")

        label_path = os.path.join(label_dir, label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects = info_label["objects"]
        texts = [
                f"{obj['detec_sensor']} | {obj['label']}"
                for obj in objects
            ]
        lidar_idx = get_lidar_idx(info_label,lidar_type)
        lidar_path = get_lidar_path(lidar_dir,lidar_type,lidar_idx)


        pcd = o3d.io.read_point_cloud(lidar_path)

   
        pcd.paint_uniform_color([0, 0, 1])
        bbox_lines = []
        text_geometries = []
        if len(objects) > 0:
            boxes = torch.stack([d["box"] for d in objects], dim=0)
            lidar_corners = boxes_to_corners_3d(boxes)
            bbox_lines = draw_bbx_lines(lidar_corners)

            if show_texts:
                text_geometries = create_bbx_text_geometries(
                    lidar_corners,
                    texts=texts,
                    z_offset=0.3,
                    outside_offset=1.0,
                    text_scale=0.1,
                    text_color=(1.0, 0.0, 0.0),
                    rotate_deg=-90
                )
            else:
                text_geometries = []

        for geo in old_geometries:
            vis.remove_geometry(geo, reset_bounding_box=False)

        current_geometries = [axis, pcd] + bbox_lines + text_geometries

        # first frame should reset bounding box
        reset_flag = need_reset
        need_reset = False

        for geo in current_geometries:
            vis.add_geometry(geo, reset_bounding_box=reset_flag)

        old_geometries = current_geometries

        view_control = vis.get_view_control()
        view_control.set_lookat(fixed_center)
        view_control.set_front([0, 0, 1])
        view_control.set_up([1, 0, 0])
        view_control.set_zoom(0.1)

        render_option = vis.get_render_option()

        render_option.background_color = np.array([1, 1, 1])
        render_option.point_size = 1.0

        vis.poll_events() 
        vis.update_renderer()

        time.sleep(frame_interval)

        frame_idx += 1

    vis.destroy_window()


# Camera Visualization

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
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )


def visualize_bbx_on_camera(camera_2d_points, image,valid_mask,texts=None,show_texts=True):

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

            cv2.line(image_with_bbx, (x1, y1), (x2, y2), (0, 0, 255), 1,cv2.LINE_AA)

        if show_texts and texts is not None and i < len(texts):
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
        label_dir,
        label_files,
        camera_dir,
        path_calib,
        start_frame_idx=0,
        max_frames=None,
        step=1,
        fps=10,
        wait_each_frame=False,
        show_texts=True
    ):

    delay = int(1000 / fps)
    step = max(1, step)

    if max_frames is None:
        max_frames = len(label_files)
    else:
        max_frames = min(max_frames, len(label_files))

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    for frame_idx in range(start_frame_idx, max_frames, step):
        print(f"frame_idx = {frame_idx}")

        label_path = os.path.join(label_dir, label_files[frame_idx])
        info_label = read_info_label(label_path)

        objects = info_label['objects']
        cam_front_idx = info_label['cam_front_idx']
        texts = [f"{obj['detec_sensor']} | {obj['label']}"for obj in objects]
        
        K, distortion, R, T = load_full_camera_calib(path_calib)
        
        camera_path = get_camera_path(camera_dir, cam_front_idx)
        camera_img = cv2.imread(camera_path, cv2.IMREAD_COLOR)
        
        if len(objects) == 0:
            image_with_bbx = camera_img
        else:
            boxes = torch.stack([d['box'] for d in objects], dim=0)
            
            lidar_corners = boxes_to_corners_3d(boxes)
            camera_corners = transform_lidar_to_camera(lidar_corners,T,R)

            img_undistort = undistort_image(
                camera_img,
                K=K,
                distortion=distortion,
                )
            
            camera_2d_points,valid_mask=camera_corners_to_2d_undistort(camera_corners,K)
            image_with_bbx=visualize_bbx_on_camera(
                camera_2d_points,
                img_undistort,
                valid_mask,
                texts,
                show_texts
            )

        cv2.imshow("camera", image_with_bbx)

        if wait_each_frame:
            key = cv2.waitKey(0) & 0xFF
        else:
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


# Radar RA Map Visualization

def visualize_bbx_on_ra_polar(ax,
                        ra_map, 
                        rae_corners,
                        arr_range, 
                        arr_azimuth_deg,
                        frame_idx,
                        texts):
    ax.clear()
    
    ra_map = np.log10(ra_map + 1e-6)

    ax.imshow(ra_map,
               origin='lower', 
               aspect='auto', 
               cmap='jet',
               extent=[arr_azimuth_deg[0], arr_azimuth_deg[-1], arr_range[0], arr_range[-1]])

    bbxes_2d = get_ra_bbx_2d(rae_corners)
    for box_idx,bbx_2d in enumerate(bbxes_2d):
        ax.plot(
            bbx_2d[:, 0],
            bbx_2d[:, 1],
            color="r",
            linewidth=2
        )
        if texts is not None and box_idx < len(texts):
            text = texts[box_idx]

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


def play_ra_polar_frames(
        frames,
        fps=10,
        window_name="RA map with bounding boxes",
        save_path=None,
        wait_each_frame=False
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

        if wait_each_frame and writer is None:
            print("Press any key for next frame, q/ESC to quit.")
            key = cv2.waitKey(0) & 0xFF
        elif num_frames == 1 and writer is None:
            print("Single frame. Press any key to close, q/ESC to quit.")
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:
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
        ], dim=1)

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

    x_min, x_max, y_min, y_max = get_ra_cartesian_limits(arr_range,arr_azimuth_deg)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max+10)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Radar y")
    ax.set_ylabel("Radar x")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")


def play_ra_cartesian_frames(
        frames,
        fps=10,
        window_name="RA map in cartesian with bounding boxes",
        save_path=None,
        wait_each_frame=False
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

        if wait_each_frame and writer is None:
            print("Press any key for next frame, q/ESC to quit.")
            key = cv2.waitKey(0) & 0xFF
        elif num_frames == 1 and writer is None:
            print("Single frame. Press any key to close, q/ESC to quit.")
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(delay) & 0xFF

        if key == ord("q") or key == 27:
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
        ], dim=1)

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

    x_min, x_max, y_min, y_max = get_ra_cartesian_limits(arr_range,arr_azimuth_deg)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max+10)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Radar y")
    ax.set_ylabel("Radar x")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")


def fig_to_cv2_image(fig):
    fig.canvas.draw()

    img_rgb = np.asarray(fig.canvas.buffer_rgba())
    img_rgb = img_rgb[:, :, :3]

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    return img_bgr


def play_ra_frames_by_step(
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        max_frames,
        R_l2r,
        T_l2r,
        radar_mode,
        start_frame_idx=0,
        step=1,
        show_texts=True,
        window_name="RA map with bounding boxes"
    ):

    step = max(1, step)

    for frame_idx in range(start_frame_idx, max_frames, step):
        print(f"Showing frame_idx = {frame_idx}")

        label_path = os.path.join(label_dir, label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects = info_label["objects"]
        tesseract_idx = info_label["tesseract_idx"]
        texts = [
            f"{obj['detec_sensor']} | {obj['label']}"
            for obj in objects
        ]
        if not show_texts:
            texts = None

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
            rae_corners = cartesian_to_rae(radar_corners)
        else:
            radar_corners = torch.zeros((0, 8, 3), dtype=torch.float32)
            rae_corners = np.zeros((0, 8, 3), dtype=np.float32)
            texts = None

        if radar_mode == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            visualize_bbx_on_ra_polar(
                ax,
                ra_map,
                rae_corners,
                arr_range,
                arr_azimuth_deg,
                frame_idx,
                texts=texts
            )

        elif radar_mode == 1 or radar_mode == 2:
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

            if radar_mode == 1:
                visualize_bbx_on_ra_cartesian(
                    ax,
                    ra_map,
                    radar_corners,
                    arr_range,
                    arr_azimuth_deg,
                    frame_idx,
                    texts=texts
                )
            else:
                visualize_bbx_on_ra_cartesian_with_yaw(
                    ax,
                    ra_map,
                    radar_corners,
                    arr_range,
                    arr_azimuth_deg,
                    frame_idx,
                    texts=texts
                )

        else:
            raise ValueError(f"Unknown radar_mode: {radar_mode}")

        image = fig_to_cv2_image(fig)
        plt.close(fig)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)

        print("Press any key or close the window for next frame, q/ESC to quit.")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return
            if key != 255:
                break
            try:
                window_closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                window_closed = True

            if window_closed:
                break

    cv2.destroyAllWindows()


def preload_ra_polar_frames(    
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        max_frames,
        R_l2r,
        T_l2r,
        start_frame_idx=0,
        show_texts=True
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
        if not show_texts:
            texts = None

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
            rae_corners = cartesian_to_rae(radar_corners)

        else:
            rae_corners = np.zeros((0, 8, 3), dtype=np.float32)
            texts=None

        fig, ax = plt.subplots(figsize=(8, 6))
        
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


def preload_ra_cartesian_frames(
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        max_frames,
        R_l2r,
        T_l2r,
        start_frame_idx=0,
        with_yaw=False,
        show_texts=True
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
        if not show_texts:
            texts = None

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

        if with_yaw:
            visualize_bbx_on_ra_cartesian_with_yaw(
                ax,
                ra_map,
                radar_corners,
                arr_range,
                arr_azimuth_deg,
                frame_idx,
                texts=texts
            )
        else:
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


def preload_ra_cartesian_frames_with_yaw(*args, **kwargs):
    return preload_ra_cartesian_frames(*args, **kwargs, with_yaw=True)


# Combined Sensor Visualization

def get_camera_frame(
        label_dir,
        label_files,
        camera_dir,
        path_calib,
        frame_idx,
        show_texts=True
    ):

    label_path = os.path.join(label_dir, label_files[frame_idx])
    info_label = read_info_label(label_path)

    objects = info_label["objects"]
    cam_front_idx = info_label["cam_front_idx"]
    texts = [
        f"{obj['detec_sensor']} | {obj['label']}"
        for obj in objects
    ]

    K, distortion, R, T = load_full_camera_calib(path_calib)

    camera_path = get_camera_path(camera_dir, cam_front_idx)
    camera_img = cv2.imread(camera_path, cv2.IMREAD_COLOR)

    if camera_img is None:
        raise FileNotFoundError(camera_path)

    img_undistort = undistort_image(
        camera_img,
        K=K,
        distortion=distortion
    )

    if len(objects) == 0:
        return img_undistort

    boxes = torch.stack([obj["box"] for obj in objects], dim=0)
    lidar_corners = boxes_to_corners_3d(boxes)
    camera_corners = transform_lidar_to_camera(lidar_corners, T, R)
    camera_2d_points, valid_mask = camera_corners_to_2d_undistort(
        camera_corners,
        K
    )

    return visualize_bbx_on_camera(
        camera_2d_points,
        img_undistort,
        valid_mask,
        texts,
        show_texts
    )


def get_lidar_frame(
        vis,
        label_dir,
        label_files,
        lidar_dir,
        lidar_type,
        frame_idx,
        old_geometries,
        show_texts=True
    ):

    label_path = os.path.join(label_dir, label_files[frame_idx])
    info_label = read_info_label(label_path)

    objects = info_label["objects"]
    texts = [
        f"{obj['detec_sensor']} | {obj['label']}"
        for obj in objects
    ]

    lidar_idx = get_lidar_idx(info_label, lidar_type)
    lidar_path = get_lidar_path(lidar_dir, lidar_type, lidar_idx)
    pcd = o3d.io.read_point_cloud(lidar_path)
    pcd.paint_uniform_color([0, 0, 1])

    reset_flag = len(old_geometries) == 0

    for geo in old_geometries:
        vis.remove_geometry(geo, reset_bounding_box=False)

    old_geometries.clear()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    current_geometries = [axis, pcd]

    if len(objects) > 0:
        boxes = torch.stack([obj["box"] for obj in objects], dim=0)
        lidar_corners = boxes_to_corners_3d(boxes)
        current_geometries.extend(draw_bbx_lines(lidar_corners))

        if show_texts:
            current_geometries.extend(
                create_bbx_text_geometries(
                    lidar_corners,
                    texts=texts,
                    z_offset=0.3,
                    outside_offset=1.0,
                    text_scale=0.1,
                    text_color=(1.0, 0.0, 0.0),
                    rotate_deg=-90
                )
            )

    for geo in current_geometries:
        vis.add_geometry(geo, reset_bounding_box=reset_flag)
        old_geometries.append(geo)

    view_control = vis.get_view_control()
    view_control.set_lookat([20.0, 0.0, 0.0])
    view_control.set_front([0, 0, 1])
    view_control.set_up([1, 0, 0])
    view_control.set_zoom(0.1)

    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])
    render_option.point_size = 1.0

    vis.poll_events()
    vis.update_renderer()

    lidar_img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    lidar_img = (lidar_img * 255).astype(np.uint8)
    lidar_img = cv2.cvtColor(lidar_img, cv2.COLOR_RGB2BGR)

    return lidar_img


def combine_sensor_frames(camera_frame, lidar_frame, radar_frame):  #from here to change the relative position
    radar_frame = cv2.resize(radar_frame, (480, 360))
    lidar_frame = cv2.resize(lidar_frame, (480, 360))
    camera_frame = cv2.resize(camera_frame, (480, 360))

    return np.hstack([
        radar_frame,
        lidar_frame,
        camera_frame
    ])


def preload_radar_frames_for_mode(
        radar_mode,
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        max_frames,
        R_l2r,
        T_l2r,
        start_frame_idx=0,
        show_texts=True
    ):

    if radar_mode == 0:
        return preload_ra_polar_frames(
            label_dir,
            label_files,
            radar_dataset,
            arr_range,
            arr_azimuth_deg,
            max_frames,
            R_l2r,
            T_l2r,
            start_frame_idx,
            show_texts
        )

    if radar_mode == 1:
        return preload_ra_cartesian_frames(
            label_dir,
            label_files,
            radar_dataset,
            arr_range,
            arr_azimuth_deg,
            max_frames,
            R_l2r,
            T_l2r,
            start_frame_idx,
            show_texts=show_texts
        )

    if radar_mode == 2:
        return preload_ra_cartesian_frames_with_yaw(
            label_dir,
            label_files,
            radar_dataset,
            arr_range,
            arr_azimuth_deg,
            max_frames,
            R_l2r,
            T_l2r,
            start_frame_idx,
            show_texts=show_texts
        )

    raise ValueError(f"Unknown radar_mode: {radar_mode}")


def get_radar_frame(
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        R_l2r,
        T_l2r,
        radar_mode,
        frame_idx,
        show_texts=True
    ):

    label_path = os.path.join(label_dir, label_files[frame_idx])
    info_label = read_info_label(label_path)

    objects = info_label["objects"]
    tesseract_idx = info_label["tesseract_idx"]
    texts = [
        f"{obj['detec_sensor']} | {obj['label']}"
        for obj in objects
    ]

    if not show_texts:
        texts = None

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
        rae_corners = cartesian_to_rae(radar_corners)
    else:
        radar_corners = torch.zeros((0, 8, 3), dtype=torch.float32)
        rae_corners = np.zeros((0, 8, 3), dtype=np.float32)
        texts = None

    if radar_mode == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        visualize_bbx_on_ra_polar(
            ax,
            ra_map,
            rae_corners,
            arr_range,
            arr_azimuth_deg,
            frame_idx,
            texts=texts
        )

    elif radar_mode == 1 or radar_mode == 2:
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
        fig.text(0.5, 0.96, title, color="white", ha="center", va="center", fontsize=12)

        if radar_mode == 1:
            visualize_bbx_on_ra_cartesian(
                ax,
                ra_map,
                radar_corners,
                arr_range,
                arr_azimuth_deg,
                frame_idx,
                texts=texts
            )
        else:
            visualize_bbx_on_ra_cartesian_with_yaw(
                ax,
                ra_map,
                radar_corners,
                arr_range,
                arr_azimuth_deg,
                frame_idx,
                texts=texts
            )

    else:
        raise ValueError(f"Unknown radar_mode: {radar_mode}")

    image = fig_to_cv2_image(fig)
    plt.close(fig)

    return image


def play_all_sensors_video(
        cfg,
        label_dir,
        label_files,
        camera_dir,
        path_calib,
        lidar_dir,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        R_l2r,
        T_l2r
    ):

    start_frame_idx = cfg.start_frame_idx

    if cfg.max_frames is None:
        max_frames = len(label_files)
    else:
        max_frames = min(cfg.max_frames, len(label_files))

    step = max(1, cfg.step)
    if cfg.all_sensors_mode == 0:
        frame_indices = list(range(start_frame_idx, max_frames, step))
        radar_frames = None
    elif cfg.all_sensors_mode == 1:
        frame_indices = list(range(start_frame_idx, max_frames, step))
        radar_frames = preload_radar_frames_for_mode(
            cfg.radar_mode,
            label_dir,
            label_files,
            radar_dataset,
            arr_range,
            arr_azimuth_deg,
            max_frames,
            R_l2r,
            T_l2r,
            cfg.start_frame_idx,
            cfg.show_texts
        )
    else:
        raise ValueError(f"Unknown all_sensors_mode: {cfg.all_sensors_mode}")

    wait_each_frame = cfg.all_sensors_mode == 0

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="LiDAR renderer",
        width=720,
        height=720,
        visible=False
    )

    old_geometries = []
    delay = int(1000 / cfg.fps)
    writer = None

    window_name = "Radar + LiDAR + Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if cfg.all_sensors_mode == 1:
        expected_duration = len(frame_indices) / cfg.fps
        print(
            f"Saving/playing {len(frame_indices)} combined frames "
            f"at {cfg.fps} FPS, duration about {expected_duration:.1f} seconds."
        )

    try:
        for frame_idx in frame_indices:
            print(f"Playing frame_idx = {frame_idx}")

            if radar_frames is None:
                radar_frame = get_radar_frame(
                    label_dir=label_dir,
                    label_files=label_files,
                    radar_dataset=radar_dataset,
                    arr_range=arr_range,
                    arr_azimuth_deg=arr_azimuth_deg,
                    R_l2r=R_l2r,
                    T_l2r=T_l2r,
                    radar_mode=cfg.radar_mode,
                    frame_idx=frame_idx,
                    show_texts=cfg.show_texts
                )
            else:
                radar_frame = radar_frames[frame_idx - start_frame_idx]

            lidar_frame = get_lidar_frame(
                vis=vis,
                label_dir=label_dir,
                label_files=label_files,
                lidar_dir=lidar_dir,
                lidar_type=cfg.lidar_type,
                frame_idx=frame_idx,
                old_geometries=old_geometries,
                show_texts=cfg.show_texts
            )

            camera_frame = get_camera_frame(
                label_dir=label_dir,
                label_files=label_files,
                camera_dir=camera_dir,
                path_calib=path_calib,
                frame_idx=frame_idx,
                show_texts=cfg.show_texts
            )

            combined = combine_sensor_frames(
                camera_frame,
                lidar_frame,
                radar_frame
            )

            if writer is None and cfg.all_sensors_save_path != "":
                h, w = combined.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    cfg.all_sensors_save_path,
                    fourcc,
                    cfg.fps,
                    (w, h)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Cannot open video writer: {cfg.all_sensors_save_path}"
                    )

            if writer is not None:
                writer.write(combined)

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, combined)

            if wait_each_frame:
                print("Press any key or close the window for next frame, q/ESC to quit.")
                while True:
                    key = cv2.waitKey(100) & 0xFF

                    if key == ord("q") or key == 27:
                        return

                    if key != 255:
                        break

                    try:
                        window_closed = cv2.getWindowProperty(
                            window_name,
                            cv2.WND_PROP_VISIBLE
                        ) < 1
                    except cv2.error:
                        window_closed = True

                    if window_closed:
                        break
            else:
                key = cv2.waitKey(delay) & 0xFF

                if key == ord("q") or key == 27:
                    break

                if key == ord(" "):
                    while True:
                        key2 = cv2.waitKey(0) & 0xFF

                        if key2 == ord(" "):
                            break

                        if key2 == ord("q") or key2 == 27:
                            return

    finally:
        if writer is not None:
            writer.release()
        vis.destroy_window()
        cv2.destroyAllWindows()
