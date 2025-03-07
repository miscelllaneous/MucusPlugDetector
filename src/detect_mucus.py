import os
import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize_3d
from skimage.morphology import binary_erosion
import multiprocessing as mp
from scipy import ndimage
from skimage.filters import threshold_yen


def paint_cross_section(point, high_intensity_area, airway, paint_mask, axis_size=5, threshold=0.95):
    x, y, z = point
    if high_intensity_area[x, y, z] == 0:
        return 0
    xy_plane_airway = airway[x-axis_size:x+axis_size+1, 
                          y-axis_size:y+axis_size+1, z]
    xy_plane_high_intensity_area = high_intensity_area[x-axis_size:x+axis_size+1, 
                          y-axis_size:y+axis_size+1, z]
    if np.sum(xy_plane_airway) == 0:
        xy_ratio = 0
    else:
        xy_ratio = np.sum(xy_plane_high_intensity_area) / np.sum(xy_plane_airway)
        
    if xy_ratio >= threshold:
        paint_mask[x-axis_size:x+axis_size+1, 
                          y-axis_size:y+axis_size+1, z] = xy_plane_high_intensity_area
    yz_plane_airway = airway[x, 
                          y-axis_size:y+axis_size+1,
                          z-axis_size:z+axis_size+1]
    yz_plane_high_intensity_area = high_intensity_area[x, 
                          y-axis_size:y+axis_size+1,
                          z-axis_size:z+axis_size+1]
    if np.sum(yz_plane_airway) == 0:
        yz_ratio = 0
    else:
        yz_ratio = np.sum(yz_plane_high_intensity_area) / np.sum(yz_plane_airway)
    if yz_ratio >= threshold:
        paint_mask[x, 
                          y-axis_size:y+axis_size+1,
                          z-axis_size:z+axis_size+1] = yz_plane_high_intensity_area
    
    xz_plane_airway = airway[x-axis_size:x+axis_size+1,
                          y,
                          z-axis_size:z+axis_size+1]
    xz_plane_high_intensity_area = high_intensity_area[x-axis_size:x+axis_size+1,
                          y,
                          z-axis_size:z+axis_size+1]
    if np.sum(xz_plane_airway) == 0:
        xz_ratio = 0
    else:
        xz_ratio = np.sum(xz_plane_high_intensity_area) / np.sum(xz_plane_airway)
    if xz_ratio >= threshold:
        paint_mask[x-axis_size:x+axis_size+1,
                          y,
                          z-axis_size:z+axis_size+1] = xz_plane_high_intensity_area
    return max(xy_ratio, yz_ratio, xz_ratio)

    
    
def process_single_case(input_img_path, input_mask_path, output_dir):

    raw_img = nib.load(input_img_path)
    affine = raw_img.affine
    img = raw_img.get_fdata()
    mask = nib.load(input_mask_path).get_fdata()

    skeleton = skeletonize_3d(mask > 0)
    binary_mask = (mask > 0).astype(int)
    
    masked_image = img * binary_mask
    masked_image[binary_mask == 0] = -1023

    skeleton = skeletonize_3d(mask > 0)
    binary_mask = (mask > 0).astype(int)
    eroded_mask = binary_erosion(binary_mask)
    valid_pixels = masked_image[eroded_mask > 0]
    yen_thresh = threshold_yen(valid_pixels)
    binary = (masked_image > yen_thresh).astype(np.int8)

    skeleton_points = np.where(skeleton > 0)

    paint_mask = np.zeros_like(mask)

    for x, y, z in zip(*skeleton_points):
        paint_cross_section((x, y, z), binary, binary_mask, paint_mask, axis_size=5, threshold=0.95)

    labeled_array, num_features = ndimage.label(paint_mask)
    component_sizes = np.bincount(labeled_array.ravel())
    
    voxel_size = np.abs(np.diag(affine)[:3])
    voxel_volume = np.prod(voxel_size) 
    
    min_volume_voxels = 1 / voxel_volume

    too_small = component_sizes < min_volume_voxels

    too_small_mask = too_small[labeled_array]
    paint_mask[too_small_mask] = 0

    basename = os.path.splitext(os.path.splitext(os.path.basename(input_img_path))[0])[0]
    output_path = os.path.join(output_dir, f"{basename}.nii.gz")
    nib.save(nib.Nifti1Image(paint_mask, affine), output_path)

def process_directory(img_dir, mask_dir, output_dir, num_processes=None):
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    process_list = []
    for filename in os.listdir(img_dir):
        if filename.endswith(".nii.gz"):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            if os.path.exists(mask_path):
                process_list.append((img_path, mask_path, output_dir))
            else:
                print(f"Warning: No matching mask found for {filename}")

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(process_single_case, process_list)

if __name__ == "__main__":
    img_dir = "inference_input"
    mask_dir = "inference_airway" 
    output_dir = "inference_mucus" 
    os.makedirs(output_dir, exist_ok=True)
    
    process_directory(img_dir, mask_dir, output_dir, num_processes=None)
