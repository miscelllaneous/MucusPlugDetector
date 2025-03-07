import os

import pytorch_lightning as pl
import torch
import tqdm
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import VNet
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    SaveImage,
    ScaleIntensityRange,
    Spacing,
)
import torch.multiprocessing as mp


class VNetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2
        )

    def forward(self, x):
        return self.model(x)


def inference_worker(gpu_id, model_path, input_files, output_directory_path, world_size, pixdim=(0.5, 0.5, 0.5)):
    device = torch.device(f"cuda:{gpu_id}")
    
    chunk_size = len(input_files) // world_size
    start_idx = gpu_id * chunk_size
    end_idx = start_idx + chunk_size if gpu_id < world_size - 1 else len(input_files)
    device_input_files = input_files[start_idx:end_idx]

    model = VNetModule()
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)

    inference_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Spacing(pixdim=pixdim, mode="bilinear"),
            ScaleIntensityRange(
                a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )


    dataset = Dataset(device_input_files, transform=inference_transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4) 

    os.makedirs(output_directory_path, exist_ok=True)

    for idx, batch in enumerate(tqdm.tqdm(dataloader)):
        input_image = batch.to(device)

        with torch.no_grad():
            roi_size = (128, 128, 128)
            sw_batch_size = 8
            output = sliding_window_inference(
                input_image,
                roi_size,
                sw_batch_size,
                model,
                overlap=0.5,
                progress=True,
                device="cpu",
                sw_device=device,
            )

        output = torch.softmax(output, dim=1)
        probabilities = output[:, 1] 
        probs = [0.5]
        for prob in probs:
            output = torch.where(probabilities > prob, torch.ones_like(probabilities), torch.zeros_like(probabilities))

            _, original_meta = LoadImage(image_only=False)(input_files[idx])
            affine = original_meta["affine"]
            output = output[0]
            SaveImage(
                output_dir=output_directory_path,
                output_postfix="",
                output_ext=".nii.gz",
                resample=True,
                separate_folder=False,
            )(output, meta_data={"affine": affine})


def inference_cpu(model_path, input_files, output_directory_path, pixdim=(0.5, 0.5, 0.5)):
    device = torch.device("cpu")
    
    model = VNetModule()
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)

    inference_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Spacing(pixdim=pixdim, mode="bilinear"),
            ScaleIntensityRange(
                a_min=-1000, a_max=500, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )

    dataset = Dataset(input_files, transform=inference_transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    os.makedirs(output_directory_path, exist_ok=True)

    for idx, batch in enumerate(tqdm.tqdm(dataloader)):
        input_image = batch.to(device)

        with torch.no_grad():
            roi_size = (128, 128, 128)
            sw_batch_size = 4 
            output = sliding_window_inference(
                input_image,
                roi_size,
                sw_batch_size,
                model,
                overlap=0.5,
                progress=True,
                device=device,
                sw_device=device,
            )

        output = torch.softmax(output, dim=1)
        probabilities = output[:, 1]
        probs = [0.5]
        for prob in probs:
            output = torch.where(probabilities > prob, torch.ones_like(probabilities), torch.zeros_like(probabilities))

            _, original_meta = LoadImage(image_only=False)(input_files[idx])
            affine = original_meta["affine"]
            output = output[0]
            SaveImage(
                output_dir=output_directory_path,
                output_postfix="",
                output_ext=".nii.gz",
                resample=True,
                separate_folder=False,
            )(output, meta_data={"affine": affine})

def inference(model_path, input_files, output_directory_path, pixdim=(0.5, 0.5, 0.5)):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        mp.spawn(
            inference_worker,
            args=(model_path, input_files, output_directory_path, num_gpus, pixdim),
            nprocs=num_gpus,
            join=True
        )
    else:
        inference_cpu(model_path, input_files, output_directory_path, pixdim)


if __name__ == "__main__":
    
    model_path = "models/two-color.pth"
    output_dir = "inference_airway"
    input_dir = "inference_input"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    input_files = [os.path.join(input_dir, file) for file in file_list]
    
    print(f"input_files: {len(input_files)}")
    print(input_files)

    inference(model_path, input_files, output_dir)
