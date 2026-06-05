import path_setup

from zxy_config import DataConfig
from zxy_label_utils import *
from zxy_data_path import *
from sensor_transformation import *
from visualization import *
from dummy_dataset import KRadarDataset


if __name__ == "__main__":

    cfg = DataConfig()

    label_dir = get_label_dir(cfg)
    label_files = get_label_files(label_dir)

    camera_dir = get_camera_dir(cfg)
    path_calib = get_camera_calib_path(cfg)

    lidar_dir = get_lidar_dir(cfg)

    radar_dir = get_radar_dir(cfg)
    radar_dataset = KRadarDataset(radar_dir)

    info_array_path = get_info_array_path(cfg)
    arr_range, arr_azimuth_deg, arr_elevation_deg = load_axis_from_mat(
        info_array_path
    )

    R_l2r, T_l2r = load_lidar2radar_calib(cfg.lidar2radar_calib_path)

    play_all_sensors_video(
        cfg=cfg,
        label_dir=label_dir,
        label_files=label_files,
        camera_dir=camera_dir,
        path_calib=path_calib,
        lidar_dir=lidar_dir,
        radar_dataset=radar_dataset,
        arr_range=arr_range,
        arr_azimuth_deg=arr_azimuth_deg,
        R_l2r=R_l2r,
        T_l2r=T_l2r
    )
