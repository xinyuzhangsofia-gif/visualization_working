import numpy as np
import torch
from collections import defaultdict

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


def read_gt_txt(gt_txt_path):
    gt_by_os2_idx= defaultdict(list)

    with open(gt_txt_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        if line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(",")]

        if len(parts) != 10:
            raise ValueError(f"Expected 10 values in gt line, got {len(parts)}: {line}")

        gt_frame_idx = int(parts[0])
        os2_64_idx = gt_frame_idx

        object_label = int(parts[1])

        a_idx = float(parts[2])
        r_idx = float(parts[3])
        a_width = float(parts[4])
        r_width = float(parts[5])
        e_idx = float(parts[6])
        e_width = float(parts[7])
        yaw = float(parts[8])
        yaw_rad = yaw * np.pi / 180.0
        cls = parts[9]

        box_rae = torch.tensor(
            [
                r_idx,
                a_idx,
                e_idx,
                r_width,
                a_width,
                e_width,
                yaw_rad
            ],
            dtype=torch.float32
        )

        obj = {
            "gt_frame_idx": gt_frame_idx, 
            "os2_64_idx":os2_64_idx,
            "object_label":object_label,
            "cls": cls,
            "class_id": object_label,
            "box_rae": box_rae,
            "raw": {
                "a_idx":a_idx,
                "r_idx": r_idx,
                "a_width": a_width,
                "r_width": r_width,
                "e_idx": e_idx,
                "e_width": e_width,
                "yaw": yaw,
                "yaw_rad": yaw_rad,
            }
        }
        gt_by_os2_idx[os2_64_idx].append(obj)
    return gt_by_os2_idx
