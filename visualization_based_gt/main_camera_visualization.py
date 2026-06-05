import path_setup

from zxy_config import DataConfig
from zxy_label_utils import *
from zxy_data_path import *
from sensor_transformation import *
from visualization import *


if __name__ == "__main__":

    cfg = DataConfig()

    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    camera_dir = get_camera_dir(cfg)
    path_calib = get_camera_calib_path(cfg)

    if cfg.camera_mode == 0:
        play_camera_video(
            label_dir,
            label_files,
            camera_dir,
            path_calib,
            start_frame_idx=cfg.start_frame_idx,
            max_frames=cfg.max_frames,
            step=cfg.step,
            fps=cfg.fps,
            wait_each_frame=True,
            show_texts=cfg.show_texts
        )

    elif cfg.camera_mode == 1:
        play_camera_video(
            label_dir,
            label_files,
            camera_dir,
            path_calib,
            start_frame_idx=cfg.start_frame_idx,
            max_frames=cfg.max_frames,
            step=cfg.step,
            fps=cfg.fps,
            wait_each_frame=False,
            show_texts=cfg.show_texts
        )

    else:
        raise ValueError(f"Unknown camera_mode: {cfg.camera_mode}")
