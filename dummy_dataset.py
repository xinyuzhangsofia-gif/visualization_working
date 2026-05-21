from torch.utils.data import Dataset
import math
import torch
import numpy as np
from scipy.io import loadmat
import glob
import os
from zxy_label_utils import read_gt_txt

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
            radar_dataset,
            gt_txt_path,
            class_to_idx=None,
            sequence=None
            ):
        super().__init__()
        self.radar_dataset = radar_dataset
        self.gt_by_file_idx = read_gt_txt(gt_txt_path)
        self.class_to_idx = class_to_idx
        self.sequence = sequence
        if self.sequence is None:
            self.sequence = getattr(radar_dataset, "sequence", None)
        if self.class_to_idx is None:
            self.class_to_idx = {
                "Sedan": 0,
                "Bus or Truck": 1,
                "Bicycle": 2,
                "Motorcycle": 3,
                "Pedestrian": 4,
                "Pedestrian Group": 5,
            }

    def __len__(self):
        return len(self.radar_dataset)

    def __getitem__(self, index):
        radar_data = self.radar_dataset[index]
        file_idx = radar_data["file_idx"]
        gt_frame_idx = radar_data["gt_frame_idx"]
        objects = self.gt_by_file_idx.get(file_idx, [])

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
            "file_idx": file_idx,
            "sequence": self.sequence,
            "image_id": f"{self.sequence}_{file_idx}",
            "rad_file": radar_data["rad_file"],
            "rae_file": radar_data["rae_file"],
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


class KRadarMultiSequenceGTDetectionDataset(Dataset):
    def __init__(self, sequence_datasets):
        super().__init__()
        if len(sequence_datasets) == 0:
            raise ValueError("sequence_datasets must not be empty")

        self.sequence_datasets = list(sequence_datasets)
        self.cumulative_sizes = []
        total = 0
        for dataset in self.sequence_datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _resolve_index(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} out of range for {len(self)} samples")

        dataset_idx = 0
        while index >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1

        previous_size = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = index - previous_size
        return dataset_idx, sample_idx

    def __getitem__(self, index):
        dataset_idx, sample_idx = self._resolve_index(index)
        return self.sequence_datasets[dataset_idx][sample_idx]

    def get_sequence_ranges(self):
        ranges = []
        start = 0
        for dataset, end in zip(self.sequence_datasets, self.cumulative_sizes):
            ranges.append({
                "sequence": getattr(dataset, "sequence", None),
                "start": start,
                "end": end,
                "length": end - start,
            })
            start = end
        return ranges


def detection_collate(batch):
    return {
        "rad": torch.stack([item["rad"] for item in batch], dim=0),
        "rae": torch.stack([item["rae"] for item in batch], dim=0),
        "gt_boxes": [item["gt_boxes"] for item in batch],
        "gt_boxes_raw": [item["gt_boxes_raw"] for item in batch],
        "gt_labels": [item["gt_labels"] for item in batch],
        "gt_frame_idx": [item["gt_frame_idx"] for item in batch],
        "file_idx": [item["file_idx"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "rad_file": [item["rad_file"] for item in batch],
        "rae_file": [item["rae_file"] for item in batch],
        "num_gt_before_fov": [item["num_gt_before_fov"] for item in batch],
        "num_gt_after_fov": [item["num_gt_after_fov"] for item in batch],
    }


class KRadarRADRAEDataset(Dataset):
    def __init__(self, rad_root_dir, sequence):
        self.sequence = sequence
        self.sequence_dir = os.path.join(rad_root_dir, str(sequence))
        self.rad_dir = os.path.join(self.sequence_dir, "rad")
        self.rae_dir = os.path.join(self.sequence_dir, "rae")

        rad_files_by_name = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob.glob(os.path.join(self.rad_dir, "*.npy"))
        }
        rae_files_by_name = {
            os.path.splitext(os.path.basename(path))[0]: path
            for path in glob.glob(os.path.join(self.rae_dir, "*.npy"))
        }

        shared_names = sorted(set(rad_files_by_name) & set(rae_files_by_name))
        if len(shared_names) == 0:
            raise ValueError(
                f"No matching rad/rae npy files found in {self.rad_dir} and {self.rae_dir}"
            )

        self.frame_names = shared_names
        self.rad_files = [rad_files_by_name[name] for name in self.frame_names]
        self.rae_files = [rae_files_by_name[name] for name in self.frame_names]

    def __len__(self):
        return len(self.rad_files)

    def _load_one_dataset_idx(self, dataset_idx):
        if dataset_idx < 0 or dataset_idx >= len(self):
            raise IndexError(
                f"dataset_idx {dataset_idx} out of range for {len(self)} radar frames"
            )

        rad_file = self.rad_files[dataset_idx]
        rae_file = self.rae_files[dataset_idx]
        rad = np.load(rad_file)
        rae = np.load(rae_file)

        return {
            "rad": rad,
            "rae": rae,
            "file_idx": dataset_idx,
            "gt_frame_idx": dataset_idx + 1,
            "frame_name": self.frame_names[dataset_idx],
            "rad_file": rad_file,
            "rae_file": rae_file,
        }

    def __getitem__(self, idx):
        return self._load_one_dataset_idx(idx)

    def get_by_file_idx(self, file_idx):
        if file_idx < 0 or file_idx >= len(self):
            raise KeyError(f"file_idx {file_idx} not found in sequence {self.sequence}")
        return self._load_one_dataset_idx(file_idx)


