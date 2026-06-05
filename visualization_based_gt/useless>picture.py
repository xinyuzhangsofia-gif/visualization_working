import os
import yaml
import numpy as np
from matplotlib import pyplot as plt


def load_calib_seq(root_folder):
    all_cam_numbers = {}
    all_rpys = {}
    all_ts = {}

    for seq_name in sorted(os.listdir(root_folder)):
        seq_path = os.path.join(root_folder, seq_name)

        if not os.path.isdir(seq_path):
            continue
        if not seq_name.startswith("seq_"):
            continue

        cam_numbers = []
        rpys = []
        ts = []

        for fname in sorted(os.listdir(seq_path)):
            if not fname.endswith(".yml"):
                continue

            path = os.path.join(seq_path, fname)

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            cam_numbers.append(data["cam_number"])

            rpys.append(np.array([
                data["roll_ldr2cam"],
                data["pitch_ldr2cam"],
                data["yaw_ldr2cam"]
            ], dtype=np.float32))

            ts.append(np.array([
                data["x_ldr2cam"],
                data["y_ldr2cam"],
                data["z_ldr2cam"]
            ], dtype=np.float32))

        if len(rpys) == 0:
            continue

        all_cam_numbers[seq_name] = np.array(cam_numbers, dtype=np.int32)
        all_rpys[seq_name] = np.stack(rpys, axis=0)
        all_ts[seq_name] = np.stack(ts, axis=0)

    return all_cam_numbers, all_rpys, all_ts


def load_calib_seq_v2(root_folder_v2):
    all_cam_numbers_v2 = {}
    all_rpys_v2 = {}
    all_ts_v2 = {}

    for seq_name in sorted(os.listdir(root_folder_v2)):
        seq_path_v2 = os.path.join(root_folder_v2, seq_name)

        if not os.path.isdir(seq_path_v2):
            continue
        if not seq_name.startswith("seq_"):
            continue

        cam_numbers_v2 = []
        rpys_v2 = []
        ts_v2 = []

        for fname in sorted(os.listdir(seq_path_v2)):
            if not fname.endswith(".yml"):
                continue

            path = os.path.join(seq_path_v2, fname)

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            cam_numbers_v2.append(data["cam_number"])

            rpys_v2.append(np.array([
                data["roll_ldr2cam"],
                data["pitch_ldr2cam"],
                data["yaw_ldr2cam"]
            ], dtype=np.float32))

            ts_v2.append(np.array([
                data["x_ldr2cam"],
                data["y_ldr2cam"],
                data["z_ldr2cam"]
            ], dtype=np.float32))

        if len(rpys_v2) == 0:
            continue

        all_cam_numbers_v2[seq_name] = np.array(cam_numbers_v2, dtype=np.int32)
        all_rpys_v2[seq_name] = np.stack(rpys_v2, axis=0)
        all_ts_v2[seq_name] = np.stack(ts_v2, axis=0)

    return all_cam_numbers_v2, all_rpys_v2, all_ts_v2


def load_cam1_params(root_folder):
    seq_ids = []
    xs = []
    ys = []
    zs = []
    rolls = []
    pitchs = []
    yaws = []

    for seq_name in sorted(os.listdir(root_folder)):
        seq_path = os.path.join(root_folder, seq_name)

        if not os.path.isdir(seq_path):
            continue
        if not seq_name.startswith("seq_"):
            continue

        cam1_path = os.path.join(seq_path, "cam_1.yml")
        if not os.path.exists(cam1_path):
            print(f"Warning: {cam1_path} not found")
            continue

        with open(cam1_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        seq_id = int(seq_name.split("_")[1])

        seq_ids.append(seq_id)
        xs.append(data["x_ldr2cam"])
        ys.append(data["y_ldr2cam"])
        zs.append(data["z_ldr2cam"])
        rolls.append(data["roll_ldr2cam"])
        pitchs.append(data["pitch_ldr2cam"])
        yaws.append(data["yaw_ldr2cam"])

    return {
        "seq_ids": np.array(seq_ids, dtype=np.int32),
        "x": np.array(xs, dtype=np.float32),
        "y": np.array(ys, dtype=np.float32),
        "z": np.array(zs, dtype=np.float32),
        "roll": np.array(rolls, dtype=np.float32),
        "pitch": np.array(pitchs, dtype=np.float32),
        "yaw": np.array(yaws, dtype=np.float32),
    }


def plot_one_parameter(seq_ids, values, ylabel, title):
    plt.figure(figsize=(8, 5))
    plt.plot(seq_ids, values, marker='o')
    plt.xlabel("seq number")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)

    root_folder = "/home/local/xinyu/K-Radar/resources/cam_calib/calib_seq"
    all_cam_numbers, all_rpys, all_ts = load_calib_seq(root_folder)

    root_folder_v2 = "/home/local/xinyu/K-Radar/resources/cam_calib/calib_seq_v2"
    all_cam_numbers_v2, all_rpys_v2, all_ts_v2 = load_calib_seq_v2(root_folder_v2)

    # ???????????
    for seq_name in sorted(all_rpys.keys()):
        print(f"\n===== {seq_name} =====")
        print("rpys:")
        print(all_rpys[seq_name])

        print("ts:")
        print(all_ts[seq_name])

    # ???????? seq ?? cam_1.yml
    cam1_data = load_cam1_params(root_folder)

    print("\n========== cam_1 data across all sequences ==========")
    print("seq_ids:")
    print(cam1_data["seq_ids"])

    print("x:")
    print(cam1_data["x"])

    print("y:")
    print(cam1_data["y"])

    print("z:")
    print(cam1_data["z"])

    print("roll:")
    print(cam1_data["roll"])

    print("pitch:")
    print(cam1_data["pitch"])

    print("yaw:")
    print(cam1_data["yaw"])

    # ??
    plot_one_parameter(
        cam1_data["seq_ids"],
        cam1_data["x"],
        ylabel="x_ldr2cam",
        title="cam_1: x change across sequences"
    )

    plot_one_parameter(
        cam1_data["seq_ids"],
        cam1_data["y"],
        ylabel="y_ldr2cam",
        title="cam_1: y change across sequences"
    )

    plot_one_parameter(
        cam1_data["seq_ids"],
        cam1_data["z"],
        ylabel="z_ldr2cam",
        title="cam_1: z change across sequences"
    )

    plot_one_parameter(
        cam1_data["seq_ids"],
        cam1_data["roll"],
        ylabel="roll_ldr2cam (deg)",
        title="cam_1: roll change across sequences"
    )

    plot_one_parameter(
        cam1_data["seq_ids"],
        cam1_data["pitch"],
        ylabel="pitch_ldr2cam (deg)",
        title="cam_1: pitch change across sequences"
    )

    plot_one_parameter(
        cam1_data["seq_ids"],
        cam1_data["yaw"],
        ylabel="yaw_ldr2cam (deg)",
        title="cam_1: yaw change across sequences"
    )