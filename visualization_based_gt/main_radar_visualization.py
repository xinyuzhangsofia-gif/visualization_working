import path_setup

from zxy_config import DataConfig
from zxy_label_utils import *
from zxy_data_path import *
from sensor_transformation import *
from visualization import *
from dummy_dataset import KRadarDataset

def get_radar_common_data(cfg):
    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)
    radar_dir = get_radar_dir(cfg)
    radar_dataset = KRadarDataset(radar_dir)
    info_array_path = get_info_array_path(cfg)
    arr_range, arr_azimuth_deg, arr_elevation_deg = load_axis_from_mat(info_array_path)
    R_l2r, T_l2r = load_lidar2radar_calib(cfg.lidar2radar_calib_path)

    return (
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        R_l2r,
        T_l2r
    )

def get_radar_max_frames(cfg, label_files):
    if cfg.max_frames is None:
        return len(label_files)

    return min(cfg.max_frames, len(label_files))


if __name__ == "__main__":

    cfg = DataConfig()

    (
        label_dir,
        label_files,
        radar_dataset,
        arr_range,
        arr_azimuth_deg,
        R_l2r,
        T_l2r
    ) = get_radar_common_data(cfg)

    max_frames = get_radar_max_frames(cfg, label_files)

    if cfg.radar_single_frame:
        play_ra_frames_by_step(
            label_dir,
            label_files,
            radar_dataset,
            arr_range,
            arr_azimuth_deg,
            max_frames,
            R_l2r,
            T_l2r,
            cfg.radar_mode,
            cfg.start_frame_idx,
            cfg.step,
            cfg.show_texts
        )

    else:
        if cfg.radar_mode == 0:
            frames = preload_ra_polar_frames(
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

            play_ra_polar_frames(
                frames=frames,
                fps=cfg.fps,
                save_path=cfg.radar_save_path
            )

        elif cfg.radar_mode == 1:
            frames = preload_ra_cartesian_frames(
                label_dir,
                label_files,
                radar_dataset,
                arr_range,
                arr_azimuth_deg,
                max_frames,
                R_l2r,
                T_l2r,
                cfg.start_frame_idx,
                show_texts=cfg.show_texts
            )

            play_ra_cartesian_frames(
                frames=frames,
                fps=cfg.fps,
                save_path=cfg.radar_save_path
            )

        elif cfg.radar_mode == 2:
            frames = preload_ra_cartesian_frames_with_yaw(
                label_dir,
                label_files,
                radar_dataset,
                arr_range,
                arr_azimuth_deg,
                max_frames,
                R_l2r,
                T_l2r,
                cfg.start_frame_idx,
                show_texts=cfg.show_texts
            )

            play_ra_cartesian_frames(
                frames=frames,
                fps=cfg.fps,
                save_path=cfg.radar_save_path
            )

        else:
            raise ValueError(f"Unknown radar_mode: {cfg.radar_mode}")
