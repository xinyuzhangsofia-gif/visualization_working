import path_setup

from zxy_config import DataConfig
from zxy_label_utils import *
from zxy_data_path import *
from sensor_transformation import *
from visualization import *
from dummy_dataset import KRadarDataset
if __name__ == "__main__":

    cfg = DataConfig()
    radar_dir = get_radar_dir(cfg)
    radar_dataset = KRadarDataset(radar_dir)

    if cfg.lidar_mode == 0:
        show_single_lidar_pcd(
            label_dir=get_label_dir(cfg),
            lidar_dir=get_lidar_dir(cfg),
            lidar_type=cfg.lidar_type,
            start_frame_idx=cfg.start_frame_idx,
            show_texts=cfg.show_texts
        )

    elif cfg.lidar_mode == 1:

        play_bev_lidar_video(
            label_dir=get_label_dir(cfg),
            lidar_dir=get_lidar_dir(cfg),
            lidar_type=cfg.lidar_type,
            start_frame_idx=cfg.start_frame_idx,
            fps=cfg.fps,
            show_texts=cfg.show_texts
        )

    else:
        raise ValueError(f"Unknown lidar_mode: {cfg.lidar_mode}")
