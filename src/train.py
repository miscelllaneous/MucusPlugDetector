import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.networks.nets import VNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandRotate90d,
    RandShiftIntensityd,
    RandRotated,
    RandCropByLabelClassesd
)
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


class WeightedDiceCELoss(nn.Module):
    def __init__(
        self,
        include_background: bool = False,
        to_onehot_y: bool = True,
        softmax: bool = True,
        squared_pred: bool = False,  
        lambda_dice: float = 0.5, 
    ):
        super().__init__()
        self.base_loss = DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            squared_pred=squared_pred,
        )
        self.lambda_dice = lambda_dice

    def forward(self, pred: torch.Tensor, label: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        if label.ndim == 5 and label.shape[1] == 1:
            label_for_ce = label[:, 0, ...]  # (N, D, H, W)
        else:
            label_for_ce = label 

        ce_loss_map = F.cross_entropy(pred, label_for_ce.long(), reduction="none")
        ce_loss_map = ce_loss_map * weight_map.squeeze(1)  # weight_map: (N, 1, ...) → (N, ...)
        ce_loss = ce_loss_map.mean()

        dice_loss = self._dice_with_weight(pred, label, weight_map)

        loss = self.lambda_dice * dice_loss + (1.0 - self.lambda_dice) * ce_loss
        return loss

    def _dice_with_weight(self, pred: torch.Tensor, label: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        pred_soft = F.softmax(pred, dim=1)  # (N, C, D, H, W)

        if label.ndim == 5 and label.shape[1] == 1:
            label = label[:, 0, ...]
        
        num_classes = pred.shape[1]
        label_one_hot = F.one_hot(label.long(), num_classes=num_classes)  # (N, D, H, W, C)
        label_one_hot = label_one_hot.permute(0, 4, 1, 2, 3).float()      # (N, C, D, H, W)

        pred_soft_fg = pred_soft[:, 1:, ...]
        label_fg = label_one_hot[:, 1:, ...]
        
        weight_map = weight_map.squeeze(1)  # (N, D, H, W)

        weight_map_broadcast = weight_map.unsqueeze(1)  

        intersection = torch.sum(weight_map_broadcast * pred_soft_fg * label_fg, dim=[0, 2, 3, 4])
        denominator = (
            torch.sum(weight_map_broadcast * pred_soft_fg, dim=[0, 2, 3, 4]) +
            torch.sum(weight_map_broadcast * label_fg,      dim=[0, 2, 3, 4]) +
            1e-8
        )
        dice_each_class = 1.0 - 2.0 * intersection / denominator  # (C-1,)
        dice_loss = dice_each_class.mean()
        return dice_loss
    

class VNetModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
        )
        if self.config["LAST_MODEL_PATH"] is not None:
            checkpoint = torch.load(self.config["LAST_MODEL_PATH"], map_location='cpu')
            state_dict = checkpoint['state_dict']
            model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            self.model.load_state_dict(model_state_dict)

        self.loss_function = WeightedDiceCELoss(
            include_background=False, 
            to_onehot_y=True, 
            softmax=True,
            lambda_dice=0.5,
        )
        
        self.training_step_outputs = []
    
    def on_train_epoch_start(self):
        self.training_step_outputs = []
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, weight_map = batch["image"], batch["label"], batch["weight_map"]
        logit_map = self.model(x)
        y = torch.where(y == 2, 1, y)
        weighted_loss = self.loss_function(logit_map, y, weight_map)
        loss = weighted_loss.mean()
        
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.training_step_outputs.append({"loss": loss})
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config["LR"], 
            weight_decay=self.config["WD"]
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["MAX_EPOCHS"],  
            eta_min=1e-6 
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
                "frequency": 1
            }
        }

    def on_train_epoch_end(self):
        self.training_step_outputs = []


class SaveLastModelCallback(Callback):
    def __init__(self, model_dir):
        super().__init__()
        self.model_dir = model_dir
        
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint(os.path.join(self.model_dir, "_last_epoch.ckpt"))


class GPUMemoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.max_memory = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        memory = torch.cuda.max_memory_allocated() / (1024**3)  
        self.max_memory = max(self.max_memory, memory)

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        print(
            f"Epoch {current_epoch} ended with max GPU memory usage: {self.max_memory:.2f} GB"
        )

    def on_train_end(self, trainer, pl_module):
        print(f"Training ended with max GPU memory usage: {self.max_memory:.2f} GB")


def get_data_loaders(config):
    img_dir = 'train_img05'
    mask_dir = 'train_airway05'
    weight_map_dir = 'train_weightmap05'
    img_list = os.listdir(mask_dir)

    img_path_list = [os.path.join(img_dir, img_name) for img_name in img_list]
    mask_path_list = [os.path.join(mask_dir, mask_name) for mask_name in img_list]
    weight_map_path_list = [os.path.join(weight_map_dir, weight_map_name) for weight_map_name in img_list]
    img_path_list.sort()
    mask_path_list.sort()
    weight_map_path_list.sort()

    train_files = [{"image": img_path, "label": mask_path, 'weight_map': weight_map_path, 'filename': img_name}
                  for img_path, mask_path, weight_map_path, img_name in zip(img_path_list, mask_path_list, weight_map_path_list, img_list)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label", "weight_map"]),
            EnsureChannelFirstd(keys=["image", "label", "weight_map"]),
            Orientationd(keys=["image", "label", "weight_map"], axcodes="RAS"),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=0.5,
            ),
            RandCropByLabelClassesd(
                keys=["image", "label", "weight_map"],
                label_key="label",
                spatial_size=config["ROI_SIZE"],
                ratios=config["SAMPLES_PER_CLASS"],
                num_samples=config["NUM_SAMPLES"],
                num_classes=2,
            ),
            RandRotate90d(
                keys=["image", "label", "weight_map"],
                prob=0.3,
                max_k=3,
            ),
            RandRotated(
                keys=["image", "label", "weight_map"],
                range_x=(-30, 30),
                range_y=(-30, 30),
                range_z=(-30, 30),
                prob=0.7,
                mode=("bilinear", "nearest", "nearest"),
            ),
        ]
    )

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=True, 
        num_workers=config["NUM_WORKERS"], 
        pin_memory=True
    )
    
    return train_loader


def train(config):
    torch.cuda.empty_cache()
    
    train_loader = get_data_loaders(config)
    model = VNetModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["MODEL_DIR"],
        filename="{epoch:02d}-{loss:.4f}",
        save_top_k=1,
        monitor="loss_epoch",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=config["MAX_EPOCHS"],
        callbacks=[
            checkpoint_callback,
            SaveLastModelCallback(config["MODEL_DIR"]),
            GPUMemoryCallback(),
        ],
        accelerator="gpu",
        devices=config["DEVICES"],
        strategy='ddp',
        log_every_n_steps=1,
        precision="16-mixed",
        max_steps=-1,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    config = {
        "ROI_SIZE": (128, 128, 128),
        "NUM_SAMPLES": 10,
        "SAMPLES_PER_CLASS": [1, 5],
        "BATCH_SIZE": 1,
        "NUM_WORKERS": 4,
        "NUM_WORKERS_VAL": 4,
        "MODEL_DIR": 'models',
        "MAX_EPOCHS": 10000,
        "DEVICES": [0],
        "LR": 0.0001,
        "WD": 0.0001,
        "LAST_MODEL_PATH": None
    }
    
    train(config)
