from torch.utils.data import Dataset
import math
import torch
import numpy as np
from scipy.io import loadmat
import glob
import os
from zxy_data_path import get_label_files
from zxy_label_utils import read_info_label, read_gt_txt

class KRadarDataset(Dataset):
    def __init__(self, radar_folder):
        self.files = sorted(glob.glob(os.path.join(radar_folder, "*.mat")))

        self.idx_to_file ={}
        for f in self.files:
            fname = os.path.basename(f)
            tesseract_idx = fname.split('_')[1].split('.')[0]
            self.idx_to_file[tesseract_idx] = f

    def __len__(self):
        return len(self.files)
    
    def _drea2rea(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=0)  

    def _drea2rad(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=2)  
    
    def _drea2aed(self, drea: np.ndarray) -> np.ndarray:
        return np.mean(drea, axis=1)
    
    def _rea2ra(self,rea:np.ndarray):
        return np.sum(rea,axis=1)
    
    def _rea2re(self,rea:np.array):
        return np.sum(rea, axis=2)

    
    def _load_one_file(self, file_path):
        drea = loadmat(file_path)['arrDREA']  
        drea = np.asarray(drea)
        rea = self._drea2rea(drea)

        return {
            "rea": rea,
            "rad": self._drea2rad(drea),
            "aed": self._drea2aed(drea),
            "ra_map":self._rea2ra(rea),
            "re_map":self._rea2re(rea)
        }
    
    def __getitem__(self, idx):
        return self._load_one_file(self.files[idx])
    
    def get_by_tesseract_idx(self,tesseract_idx):
        if tesseract_idx not in self.idx_to_file:
            raise KeyError(f"tesseract_idx {tesseract_idx} not found in dataset")
        file_path=self.idx_to_file[tesseract_idx]
        return self._load_one_file(file_path)


class KRadarGTDetectionDataset(Dataset):
    def __init__(
            self,
            label_dir,
            radar_dataset,
            gt_txt_path,
            label_files=None,
            class_to_idx=None
            ):
        super().__init__()
        self.label_dir = label_dir
        self.label_files = label_files
        if self.label_files is None:
            self.label_files = get_label_files(label_dir)

        self.radar_dataset = radar_dataset
        self.gt_by_frame_idx = read_gt_txt(gt_txt_path)
        self.class_to_idx = class_to_idx
        if self.class_to_idx is None:
            self.class_to_idx = {
                "Sedan": 0,
                "Bus or Truck": 1,
            }

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        label_path = f"{self.label_dir}/{self.label_files[index]}"
        info_label = read_info_label(label_path)

        gt_frame_idx = int(info_label["os2_64_idx"])
        objects = self.gt_by_frame_idx.get(gt_frame_idx, [])

        radar_data = self.radar_dataset.get_by_tesseract_idx(
            info_label["tesseract_idx"]
        )

        rad = torch.from_numpy(radar_data["rad"]).float()
        rae = torch.from_numpy(radar_data["rae"]).float()
        objects_in_fov = [
            obj for obj in objects
            if self._object_center_in_rae_fov(obj, rae.shape)
        ]

        if len(objects_in_fov) > 0:
            gt_boxes_raw = torch.stack([obj["box_rae"] for obj in objects_in_fov], dim=0)
            gt_boxes = self._normalize_boxes_rae(gt_boxes_raw, rae.shape)
            gt_labels = torch.tensor(
                [self._class_id(obj) for obj in objects_in_fov],
                dtype=torch.long
            )
        else:
            gt_boxes_raw = torch.zeros((0, 7), dtype=torch.float32)
            gt_boxes = torch.zeros((0, 7), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.long)

        return {
            "rad": rad,
            "rae": rae,
            "gt_boxes": gt_boxes,
            "gt_boxes_raw": gt_boxes_raw,
            "gt_labels": gt_labels,
            "gt_frame_idx": gt_frame_idx,
            "tesseract_idx": info_label["tesseract_idx"],
            "label_file": self.label_files[index],
            "num_gt_before_fov": len(objects),
            "num_gt_after_fov": len(objects_in_fov),
        }

    def _object_center_in_rae_fov(self, obj, rae_shape):
        r_size, a_size, e_size = rae_shape
        raw = obj["raw"]
        return (
            0 <= raw["r_idx"] < r_size
            and 0 <= raw["a_idx"] < a_size
            and 0 <= raw["e_idx"] < e_size
        )

    def _normalize_boxes_rae(self, boxes, rae_shape):
        r_size, a_size, e_size = rae_shape
        normalized = boxes.clone()
        normalized[:, 0] = normalized[:, 0] / r_size
        normalized[:, 1] = normalized[:, 1] / a_size
        normalized[:, 2] = normalized[:, 2] / e_size
        normalized[:, 3] = normalized[:, 3] / r_size
        normalized[:, 4] = normalized[:, 4] / a_size
        normalized[:, 5] = normalized[:, 5] / e_size
        normalized[:, 6] = ((normalized[:, 6] + np.pi) % (2.0 * np.pi)) / (2.0 * np.pi)
        return normalized.clamp(0.0, 1.0)

    def _class_id(self, obj):
        cls = obj["cls"]
        if cls not in self.class_to_idx:
            raise KeyError(
                f"Unknown GT class {cls!r}. Add it to class_to_idx. "
                f"Known classes: {sorted(self.class_to_idx)}"
            )
        return self.class_to_idx[cls]


def detection_collate(batch):
    return {
        "rad": torch.stack([item["rad"] for item in batch], dim=0),
        "rae": torch.stack([item["rae"] for item in batch], dim=0),
        "gt_boxes": [item["gt_boxes"] for item in batch],
        "gt_boxes_raw": [item["gt_boxes_raw"] for item in batch],
        "gt_labels": [item["gt_labels"] for item in batch],
        "gt_frame_idx": [item["gt_frame_idx"] for item in batch],
        "tesseract_idx": [item["tesseract_idx"] for item in batch],
        "label_file": [item["label_file"] for item in batch],
        "num_gt_before_fov": [item["num_gt_before_fov"] for item in batch],
        "num_gt_after_fov": [item["num_gt_after_fov"] for item in batch],
    }


class KRadarRADRAEDataset(Dataset):
    def __init__(self, rad_root_dir, sequence):
        self.sequence = sequence
        self.sequence_dir = os.path.join(rad_root_dir, str(sequence))
        self.rad_dir = os.path.join(self.sequence_dir, "rad")
        self.rae_dir = os.path.join(self.sequence_dir, "rae")

        self.rad_files = sorted(glob.glob(os.path.join(self.rad_dir, "*.npy")))
        self.rae_files = sorted(glob.glob(os.path.join(self.rae_dir, "*.npy")))

        self.idx_to_rad_file = {}
        self.idx_to_rae_file = {}

        for f in self.rad_files:
            fname = os.path.basename(f)
            tesseract_idx = os.path.splitext(fname)[0]
            self.idx_to_rad_file[tesseract_idx] = f

        for f in self.rae_files:
            fname = os.path.basename(f)
            tesseract_idx = os.path.splitext(fname)[0]
            self.idx_to_rae_file[tesseract_idx] = f

        self.tesseract_indices = sorted(
            set(self.idx_to_rad_file.keys()) & set(self.idx_to_rae_file.keys())
        )

    def __len__(self):
        return len(self.tesseract_indices)

    def _load_one_tesseract(self, tesseract_idx):
        if tesseract_idx not in self.idx_to_rad_file:
            raise KeyError(f"rad file not found for tesseract_idx {tesseract_idx}")

        if tesseract_idx not in self.idx_to_rae_file:
            raise KeyError(f"rae file not found for tesseract_idx {tesseract_idx}")

        rad = np.load(self.idx_to_rad_file[tesseract_idx])
        rae = np.load(self.idx_to_rae_file[tesseract_idx])

        return {
            "rad": rad,
            "rae": rae
        }

    def __getitem__(self, idx):
        tesseract_idx = self.tesseract_indices[idx]
        return self._load_one_tesseract(tesseract_idx)

    def get_by_tesseract_idx(self, tesseract_idx):
        return self._load_one_tesseract(tesseract_idx)

# if __name__ == "__main__":
#     dataset = KRadarRADDataset(
#         "/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar-RAD",
#         1
#     )
#     data = dataset.get_by_tesseract_idx("00033")

#     print(data["rad"].shape, data["rad"].dtype)
#     print(data["rae"].shape, data["rae"].dtype)
#     dataset = KRadarDataset("/home/local/xinyu/KRadar/1/radar_tesseract")
#     data = dataset[0]
#     print(dataset[0]['rae'].shape)
