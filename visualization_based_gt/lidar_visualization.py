import os
import path_setup
import numpy as np
import torch
import time
import open3d as o3d
from matplotlib import pyplot as plt
from lidar2radar_transformation import read_info_label,boxes_to_corners_3d
from PIL import Image, ImageDraw, ImageFont

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



def play_bev_lidar_video(label_dir, lidar_dir,lidar_type, frame_idx=0, fps=10,show_texts=True):
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="BEV LiDAR Video",
        width=720,
        height=720
    )

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    old_geometries = []

    frame_interval = 1.0 / fps

    # fixed BEV center
    fixed_center = np.array([20.0, 0.0, 0.0])

    for frame_idx in range(frame_idx, len(label_files)):
        print(f"Playing frame_idx = {frame_idx}")

        label_path = os.path.join(label_dir, label_files[frame_idx])
        info_label = read_info_label(label_path)
        objects = info_label["objects"]
        texts = [
                f"{obj['detec_sensor']} | {obj['label']}"
                for obj in objects
            ]
        os1_128_idx = info_label['os1_128_idx']
        os2_64_idx = info_label['os2_64_idx']
        if lidar_type == 'os1-128':
            lidar_idx = os1_128_idx
        elif lidar_type == 'os2-64':
            lidar_idx = os2_64_idx
        lidar_path = get_lidar_path(lidar_dir, lidar_type,lidar_idx)


        pcd = o3d.io.read_point_cloud(lidar_path)

        # important: make points visible
        pcd.paint_uniform_color([0, 0, 1])

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
        reset_flag = (frame_idx == frame_idx)

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

    vis.destroy_window()



def get_lidar_path(lidar_dir,lidar_type,lidar_idx):
    for fname in sorted(os.listdir(lidar_dir)):
        if fname.startswith(f"{lidar_type}_{lidar_idx}"):
            return os.path.join(lidar_dir,fname)
    raise FileNotFoundError(f"{lidar_type}-lidar file not found for idx{lidar_idx} in {lidar_dir}")


if __name__ == "__main__": 
    frame_idx = 0
    choose_info_label='info_label_rev2'
    lidar_mode = "bev_video"
    #lidar_mode = "single"
    lidar_type = 'os2-64' 
    sequence=1
    show_texts = False
    fps=10

    label_dir= f'/home/local/xinyu/KRadar/{sequence}/{choose_info_label}'
    label_files=sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    lidar_dir = f'/home/local/xinyu/KRadar/{sequence}/{lidar_type}'
    if lidar_mode == "single":

        for frame_idx in range(frame_idx,len(label_files)):
            print(f"frame_idx = {frame_idx}")
            label_path=os.path.join(label_dir,label_files[frame_idx])
            
            info_label = read_info_label(label_path)
            objects = info_label['objects']

            os1_128_idx = info_label['os1_128_idx']
            os2_64_idx = info_label['os2_64_idx']
            if lidar_type == 'os1-128':
                lidar_idx=os1_128_idx
            elif lidar_type == 'os2-64':
                lidar_idx = os2_64_idx
            lidar_path = get_lidar_path(lidar_dir, lidar_type,lidar_idx)

            pcd = o3d.io.read_point_cloud(lidar_path)

            boxes=torch.stack([d['box'] for d in objects],dim=0)
            lidar_corners=boxes_to_corners_3d(boxes)
            texts = [
                    f"{obj['detec_sensor']} | {obj['label']}"
                    for obj in objects
                ]
            visualize_bbx_on_lidar_pcd(lidar_corners,pcd,texts,show_texts)
    
    elif lidar_mode == "bev_video":
        play_bev_lidar_video(
            label_dir,
            lidar_dir,
            lidar_type,
            frame_idx,
            fps
        )
