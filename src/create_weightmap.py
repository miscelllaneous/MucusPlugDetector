import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d
import os
import multiprocessing as mp
from functools import partial

def create_airway_weight_map(img):

    airway_mask = img.get_fdata()
    binary_mask = (airway_mask > 0).astype(np.uint8)

    print("skeletonizing...", binary_mask.shape)
    skeleton = skeletonize_3d(binary_mask)
    reversed_skeleton = np.ones_like(skeleton)
    reversed_skeleton[skeleton == 1] = 0
    print("distance transform...", reversed_skeleton.shape)
    distance_map = np.zeros_like(binary_mask, dtype=float)
    distance_map = distance_transform_edt(reversed_skeleton)

    weight_map = np.ones_like(distance_map)

    weight_values = {
        (0, 1): 32,
        (1, 2): 16, 
        (2, 3): 8, 
        (3, 4): 4,  
        (4, 5): 2,  
    }

    for (min_dist, max_dist), weight in weight_values.items():
        mask = (distance_map >= min_dist) & (distance_map < max_dist)
        weight_map[mask] = weight
    
    output_img = nib.Nifti1Image(weight_map, img.affine)
    return output_img

def pipeline(input_path, weight_map_path="weightmap"):
    img = nib.load(input_path)
    file_name = os.path.basename(input_path)
    weight_map = create_airway_weight_map(img)
    nib.save(weight_map, os.path.join(weight_map_path, file_name))

def process_file(input_file, input_dir, weight_map_path):
    input_path = os.path.join(input_dir, input_file)
    pipeline(input_path, weight_map_path)

if __name__ == "__main__":
    input_dir = "train_input"
    weight_map_path = "train_weightmap"
    input_files = os.listdir(input_dir)
    os.makedirs(weight_map_path, exist_ok=True)
    
    num_processes = max(1, mp.cpu_count() - 1)
    
    process_func = partial(process_file, 
                         input_dir=input_dir, 
                         weight_map_path=weight_map_path)
    
    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_func, input_files)
