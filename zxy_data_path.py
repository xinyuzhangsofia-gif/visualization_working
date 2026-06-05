import os


def get_label_dir(cfg):
    label_dir = f"{cfg.root_dir}/{cfg.sequence}/{cfg.choose_info_label}"
    return label_dir

def get_label_files(label_dir):
    label_files = sorted([f for f in os.listdir(label_dir)if f.endswith(".txt")])
    return label_files


def get_lidar_dir(cfg):
    lidar_dir = f"{cfg.root_dir}/{cfg.sequence}/{cfg.lidar_type}"
    return lidar_dir


def get_lidar_idx(info_label, lidar_type):
    if lidar_type == "os1-128":
        return info_label["os1_128_idx"]

    elif lidar_type == "os2-64":
        return info_label["os2_64_idx"]

    else:
        raise ValueError(f"Unknown lidar_type: {lidar_type}")
    
    
def get_lidar_path(lidar_dir,lidar_type,lidar_idx):
    for fname in sorted(os.listdir(lidar_dir)):
        if fname.startswith(f"{lidar_type}_{lidar_idx}"):
            return os.path.join(lidar_dir,fname)
    raise FileNotFoundError(f"{lidar_type}-lidar file not found for idx{lidar_idx} in {lidar_dir}")


def get_camera_dir(cfg):
    camera_dir = f"{cfg.root_dir}/{cfg.sequence}/{cfg.choose_camera}_front"
    return camera_dir


def get_camera_path(camera_dir,cam_front_idx):
    for fname in sorted(os.listdir(camera_dir)):
        if fname.startswith(f"cam-front_{cam_front_idx}"):
            return os.path.join(camera_dir,fname)

    raise FileNotFoundError(f"cam-front file not found for idx{cam_front_idx} in {camera_dir}")


def get_camera_calib_path(cfg):
    if cfg.sequence < 10:
        path_calib = f"{cfg.root_dir}/{cfg.calib_seq}/seq_0{cfg.sequence}/{cfg.choose_camera}.yml"
    else:
        path_calib = f"{cfg.root_dir}/{cfg.calib_seq}/seq_{cfg.sequence}/{cfg.choose_camera}.yml"

    return path_calib


def get_info_array_path(cfg):
    info_array_path = f"{cfg.root_dir}/info_arr.mat"
    return info_array_path


def get_lidar2radar_calib_path(cfg):
    lidar2radar_calib_path = cfg.lidar2radar_calib_path
    return lidar2radar_calib_path

def get_radar_dir(cfg):
    radar_dir = f"/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar/{cfg.sequence}/radar_tesseract"
    return radar_dir


def get_radar_path(radar_dir,tesseract_idx):
    for fname in sorted(os.listdir(radar_dir)):
        if fname.startswith(f"tesseract_{tesseract_idx}"):
            return os.path.join(radar_dir,fname)
    raise FileNotFoundError(f"tesseract file not found for idx{tesseract_idx} in {radar_dir}")


def get_rad_rae_npy_root_dir():
    return "/home/local/xinyu/K-Radar-RAD"



def get_rad_npy_dir(cfg):
    return os.path.join(get_rad_rae_npy_root_dir(), str(cfg.sequence), "rad")


def get_rae_npy_dir(cfg):
    return os.path.join(get_rad_rae_npy_root_dir(), str(cfg.sequence), "rae")


def get_rad_npy_files(cfg):
    return sorted([
        os.path.join(get_rad_npy_dir(cfg), f)
        for f in os.listdir(get_rad_npy_dir(cfg))
        if f.endswith(".npy")
    ])


def get_rae_npy_files(cfg):
    return sorted([
        os.path.join(get_rae_npy_dir(cfg), f)
        for f in os.listdir(get_rae_npy_dir(cfg))
        if f.endswith(".npy")
    ])


def get_rad_npy_path(cfg, file_idx):
    rad_files = get_rad_npy_files(cfg)
    if file_idx < 0 or file_idx >= len(rad_files):
        raise IndexError(f"file_idx {file_idx} out of range for {len(rad_files)} rad npy files")
    return rad_files[file_idx]


def get_rae_npy_path(cfg, file_idx):
    rae_files = get_rae_npy_files(cfg)
    if file_idx < 0 or file_idx >= len(rae_files):
        raise IndexError(f"file_idx {file_idx} out of range for {len(rae_files)} rae npy files")
    return rae_files[file_idx]


def get_gt_txt_path(cfg, sequence=None):
    if sequence is None:
        sequence = cfg.sequence
    # gt_txt_path = f"/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar-GT-Polar-v2/{cfg.sequence}/gt/gt.txt"
    gt_txt_path = f"/home/local/xinyu/K-Radar-GT-Polar-v2/{sequence}/gt/gt.txt"
    return gt_txt_path
