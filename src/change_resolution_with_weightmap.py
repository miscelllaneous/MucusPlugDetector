import os
import numpy as np
from monai.transforms import LoadImaged, SaveImaged, Spacingd, Compose, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, CropForegroundd
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

def convert_resolution_for_pair(
    image_path_dict: dict,
    output_dir1: str,
    output_dir2: str,
    output_dir3: str,
    output_dir4: str,
    target_spacing: tuple = (0.5, 0.5, 0.5)
):
    spacing_transform_image = Compose(
    [
            LoadImaged(keys=["image", "mask", "lungmask", "weight_map"]),
            EnsureChannelFirstd(keys=["image", "mask", "lungmask", "weight_map"]),
            Orientationd(keys = ["image", "mask", "lungmask", "weight_map"], axcodes="RAS"),
            Spacingd(
                keys=["image", "mask", "lungmask", "weight_map"],
                pixdim=target_spacing,
                mode=["bilinear", "nearest", "nearest", "nearest"],
            ),
            CropForegroundd(keys=["image", "mask", "lungmask", "weight_map"], source_key="lungmask"),
            ScaleIntensityRanged(
               keys=["image"],
               a_min=-1000,
               a_max=500,
               b_min=0.0,
               b_max=1.0,
               clip=True,
            ),
            SaveImaged(keys=["image"],output_dir=output_dir1,output_dtype=np.float32, output_postfix='', separate_folder=False),
            SaveImaged(keys=["mask"],output_dir=output_dir2,output_dtype=np.uint8, output_postfix='', separate_folder=False),
            SaveImaged(keys=["lungmask"],output_dir=output_dir3,output_dtype=np.uint8, output_postfix='', separate_folder=False),
            SaveImaged(keys=["weight_map"],output_dir=output_dir4,output_dtype=np.float32, output_postfix='', separate_folder=False)
        ]
    )

    spacing_transform_image(image_path_dict)

def process_single_case(file_name, img_dir, mask_dir, lungmask_dir, weight_map_dir, output_dir1, output_dir2, output_dir3, output_dir4, resolution):
    img_path = os.path.join(img_dir, file_name)
    mask_path = os.path.join(mask_dir, file_name)
    lungmask_path = os.path.join(lungmask_dir, file_name)
    weight_map_path = os.path.join(weight_map_dir, file_name)

    dict = { "image": img_path , "mask": mask_path, "lungmask": lungmask_path, "weight_map": weight_map_path}

    convert_resolution_for_pair(dict, output_dir1, output_dir2, output_dir3, output_dir4, target_spacing=resolution)

if __name__ == "__main__":
    img_dir = "../train_input" 
    mask_dir = "../train_mask"
    lungmask_dir = "../train_lungmask"
    weight_map_dir = "../train_weightmap"
    output_dir1 = "../train_img05"
    output_dir2 = "../train_mask05"
    output_dir3 = "../train_lungmask05"
    output_dir4 = "../train_weightmap05"
    resolution = (0.5, 0.5, 0.5)
    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)
    os.makedirs(output_dir3, exist_ok=True)
    os.makedirs(output_dir4, exist_ok=True)

    file_names = os.listdir(mask_dir)

    process_func = partial(
        process_single_case,
        img_dir=img_dir,
        mask_dir=mask_dir,
        lungmask_dir=lungmask_dir,
        weight_map_dir=weight_map_dir,
        output_dir1=output_dir1,
        output_dir2=output_dir2,
        output_dir3=output_dir3,
        output_dir4=output_dir4,
        resolution=resolution
    )

    num_processes = min(20, len(file_names))

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_func, file_names), total=len(file_names)))