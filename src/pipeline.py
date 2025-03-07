import os
import nibabel as nib
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    ScaleIntensityRange,
    Spacing,
    LoadImage,
)
import pytorch_lightning as pl
from monai.networks.nets import VNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from scipy.ndimage import zoom
from typing import Tuple
from lungmask import LMInferer
from scipy import ndimage
import SimpleITK as sitk


class VNetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2
        )
        self.save_hyperparameters()
        self.loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=2, get_not_nans=False
        )

    def forward(self, x):
        return self.model(x)


class MedastinalMaskInference:
    def __init__(self, model_path, input_base_dir, output_base_dir, device='cpu'):
        self.device = device
        self.model = VNetModule()
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
        
        self.transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Spacing(pixdim=(2.0, 2.0, 2.0)),
            ScaleIntensityRange(
                a_min=-2000,
                a_max=500,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
        ])
        self.lungmask_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Spacing(pixdim=(2.0, 2.0, 2.0)),
        ])

        self.output_base_dir = output_base_dir
        self.input_base_dir = input_base_dir

    @torch.no_grad()
    def create_mediastinal_mask(self, image_path,) -> nib.Nifti1Image:
        output_path = os.path.join(self.output_base_dir, "mediastinal_masks", os.path.basename(image_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nii_img = nib.load(image_path)
        original_affine = nii_img.affine
        original_shape = nii_img.get_fdata().shape
        
        img_data = self.transforms(image_path)

        img_data = img_data.unsqueeze(0)
        img_data = img_data.to(self.device)
        
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        output = sliding_window_inference(
            img_data, 
            roi_size, 
            sw_batch_size, 
            self.model,
            overlap=0.6
        )
        
        output = torch.softmax(output, dim=1)
        mask = torch.argmax(output, dim=1).squeeze(0)
        mask = mask.cpu().numpy().astype(np.uint8)
        
        zoom_factors = (
            original_shape[0] / mask.shape[0],
            original_shape[1] / mask.shape[1],
            original_shape[2] / mask.shape[2]
        )
        mask = zoom(mask, zoom_factors, order=0)  
        
        mask_nii = nib.Nifti1Image(mask, original_affine)
        nib.save(mask_nii, output_path)
        return mask_nii
    
    def apply_lungmask(self, image_path: str, lung_mask_path: str) -> str:
        output_path = os.path.join(self.output_base_dir, "masked_images", os.path.basename(image_path))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img = nib.load(image_path)
        mask = nib.load(lung_mask_path)
        
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()
        
        mask_data[mask_data == 2] = 1
        
        masked_data = img_data.copy()
        masked_data[mask_data == 1] = -2000
        
        masked_img = nib.Nifti1Image(masked_data, img.affine, img.header)
        nib.save(masked_img, output_path)
        
        return output_path

    def create_lungmask(self, input_file: str) -> str:
        output_file = os.path.join(self.output_base_dir, "lungmasks", os.path.basename(input_file))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            inferer = LMInferer()
            input_image = sitk.ReadImage(input_file)
            segmentation = inferer.apply(input_image)
            segmentation_image = sitk.GetImageFromArray(segmentation)
            segmentation_image.CopyInformation(input_image)
            sitk.WriteImage(segmentation_image, output_file)
        except Exception as e:
            print(f"Error creating lung mask for {input_file}: {str(e)}")
            raise
        return output_file

    def create_inner_outer_lung_mask(self, lung_mask_img: nib.Nifti1Image, mediastinum_mask_img: nib.Nifti1Image, threshold: int = 16) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:

        ras_affine = lung_mask_img.affine
        lung_mask_array = lung_mask_img.get_fdata()
        lung_mask_original = lung_mask_array.copy()
        lung_mask_array [lung_mask_array == 2] = 1
        mediastinum_mask_array = mediastinum_mask_img.get_fdata()

        mediastinum_mask_array = ndimage.binary_dilation(mediastinum_mask_array, iterations=7)

        shrinked_mask_array = ndimage.binary_erosion(lung_mask_array, iterations=1)
        surface_mask_array = np.logical_and(lung_mask_array == 1, shrinked_mask_array == 0).astype(np.int8)
        outer_surface_mask_array = np.logical_and(surface_mask_array, mediastinum_mask_array == 0).astype(np.int8)
        
        final_outer_surface = np.zeros_like(outer_surface_mask_array)
        for lung_value in [1, 2]:
            current_lung_surface = np.logical_and(
                outer_surface_mask_array == 1,
                lung_mask_original == lung_value
            )
            
            labeled_array, num_features = ndimage.label(current_lung_surface, structure=np.ones((3,3,3)))
            if num_features > 0:
                component_sizes = [(i, np.sum(labeled_array == i)) for i in range(1, num_features + 1)]
                largest_component = max(component_sizes, key=lambda x: x[1])[0]
                final_outer_surface = np.logical_or(
                    final_outer_surface,
                    labeled_array == largest_component
                )
        
        outer_surface_mask_array = final_outer_surface.astype(np.int8)


        pixdim = lung_mask_img.header.get_zooms()
        outer_surface_mask_array_resampled = ndimage.zoom(outer_surface_mask_array, pixdim, order=0).astype(np.int8)
        lung_mask_array_resampled = ndimage.zoom(lung_mask_array, pixdim, order=0).astype(np.int8)

        distance_from_outer_surface = ndimage.distance_transform_edt(~outer_surface_mask_array_resampled.astype(bool))
        dilated_outer_surface = (distance_from_outer_surface <= threshold).astype(np.int8)
        outer_lung_mask_array_resampled = np.logical_and(lung_mask_array_resampled == 1, dilated_outer_surface == 1).astype(np.int8)
        inner_lung_mask_array_resampled = np.logical_and(lung_mask_array_resampled == 1, outer_lung_mask_array_resampled == 0).astype(np.int8)

        scale_factors = [1.0/dim for dim in pixdim]

        outer_lung_mask_array = ndimage.zoom(outer_lung_mask_array_resampled, scale_factors, order=0).astype(np.int8)
        inner_lung_mask_array = ndimage.zoom(inner_lung_mask_array_resampled, scale_factors, order=0).astype(np.int8)

        inner_lung_img = nib.Nifti1Image(inner_lung_mask_array, ras_affine)
        outer_lung_img = nib.Nifti1Image(outer_lung_mask_array, ras_affine)
        return inner_lung_img, outer_lung_img
    
    def pipeline(self):
        for filename in os.listdir(self.input_base_dir):
            if filename.endswith(".nii.gz"):
                input_path = os.path.join(self.input_base_dir, filename)
                lungmask_path = self.create_lungmask(input_path)
                lungmask_nii = nib.load(lungmask_path)
                
                masked_image_path = self.apply_lungmask(input_path, lungmask_path)
                
                mask_nii = self.create_mediastinal_mask(masked_image_path)
                
                inner_lung_img, outer_lung_img = self.create_inner_outer_lung_mask(lungmask_nii, mask_nii, threshold=20)
                os.makedirs(os.path.join(self.output_base_dir, "inner_lung_masks"), exist_ok=True)
                os.makedirs(os.path.join(self.output_base_dir, "outer_lung_masks"), exist_ok=True)
                nib.save(inner_lung_img, os.path.join(self.output_base_dir, "inner_lung_masks", filename))
                nib.save(outer_lung_img, os.path.join(self.output_base_dir, "outer_lung_masks", filename))
                if os.path.exists(lungmask_path):
                    os.remove(lungmask_path)
                if os.path.exists(masked_image_path):
                    os.remove(masked_image_path)
                mediastinum_dilated_path = os.path.join(self.output_base_dir, "mediastinum_mask_dilation.nii.gz")
                if os.path.exists(mediastinum_dilated_path):
                    os.remove(mediastinum_dilated_path)
                mediastinal_mask_path = os.path.join(self.output_base_dir, "mediastinal_masks", os.path.basename(masked_image_path))
                if os.path.exists(mediastinal_mask_path):
                    os.remove(mediastinal_mask_path)


    def get_surface_from_rays(self, mask_array: np.ndarray) -> np.ndarray:

        surface = np.zeros_like(mask_array)
        
        for axis in range(3):  
            for direction in [1, -1]:  
                temp_surface = np.zeros_like(mask_array)
                if direction == 1:
                    for i in range(mask_array.shape[axis]):
                        if axis == 0:
                            slice_mask = mask_array[i, :, :]
                            if np.any(slice_mask): 
                                temp_surface[i, :, :] = np.logical_and(
                                    slice_mask,
                                    np.logical_not(temp_surface[i-1, :, :] if i > 0 else np.zeros_like(slice_mask))
                                )
                        elif axis == 1:
                            slice_mask = mask_array[:, i, :]
                            if np.any(slice_mask):
                                temp_surface[:, i, :] = np.logical_and(
                                    slice_mask,
                                    np.logical_not(temp_surface[:, i-1, :] if i > 0 else np.zeros_like(slice_mask))
                                )
                        else:
                            slice_mask = mask_array[:, :, i]
                            if np.any(slice_mask):
                                temp_surface[:, :, i] = np.logical_and(
                                    slice_mask,
                                    np.logical_not(temp_surface[:, :, i-1] if i > 0 else np.zeros_like(slice_mask))
                                )
                else:
                    for i in range(mask_array.shape[axis]-1, -1, -1):
                        if axis == 0:
                            slice_mask = mask_array[i, :, :]
                            if np.any(slice_mask):
                                temp_surface[i, :, :] = np.logical_and(
                                    slice_mask,
                                    np.logical_not(temp_surface[i+1, :, :] if i < mask_array.shape[axis]-1 else np.zeros_like(slice_mask))
                                )
                        elif axis == 1:
                            slice_mask = mask_array[:, i, :]
                            if np.any(slice_mask):
                                temp_surface[:, i, :] = np.logical_and(
                                    slice_mask,
                                    np.logical_not(temp_surface[:, i+1, :] if i < mask_array.shape[axis]-1 else np.zeros_like(slice_mask))
                                )
                        else: 
                            slice_mask = mask_array[:, :, i]
                            if np.any(slice_mask):
                                temp_surface[:, :, i] = np.logical_and(
                                    slice_mask,
                                    np.logical_not(temp_surface[:, :, i+1] if i < mask_array.shape[axis]-1 else np.zeros_like(slice_mask))
                                )
                
                surface = np.logical_or(surface, temp_surface)
        
        return surface.astype(np.int8)

def main():
    input_dir = "inference_input" 
    output_dir = "inference_masks"  
    model_path = "models/vnet_2.pth"  
    os.makedirs(output_dir, exist_ok=True)
    
    inferencer = MedastinalMaskInference(model_path, input_dir, output_dir)
    inferencer.pipeline()

if __name__ == "__main__":
    main()