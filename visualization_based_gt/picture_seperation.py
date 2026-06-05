import os
import cv2
choose_direction = 'left'

input_dir = "/run/user/1000/gvfs/smb-share:server=192.168.189.30,share=elab-share/Datasets/K-Radar/11/cam-front"
if choose_direction == 'right':
    output_dir = "/home/local/xinyu/KRadar/11/cam_front_right"
elif choose_direction =='left':
    output_dir = "/home/local/xinyu/KRadar/11/cam_front_left"

# create output folder if not exists
os.makedirs(output_dir, exist_ok=True)

# iterate all files
for fname in sorted(os.listdir(input_dir)):

    # only process png files
    if not fname.endswith(".png"):
        continue

    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    # read image
    img = cv2.imread(input_path)

    if img is None:
        print(f"Failed to read: {input_path}")
        continue

    h, w = img.shape[:2]

    if choose_direction == 'right':
        img_crop = img[:, w//2:]     # ???
    else:
        img_crop = img[:, :w//2]     # ???
    # save
    cv2.imwrite(output_path, img_crop)

    print(f"Processed: {fname}")