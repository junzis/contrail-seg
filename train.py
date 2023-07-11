# %%
import glob
import warnings
import torch
import lightning
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import contrail

warnings.filterwarnings("ignore")

# %%
image_paths = sorted(glob.glob(f"data/train_*/image/*.png"))
mask_paths = sorted(glob.glob(f"data/train_*/mask/*.png"))

x_train, x_val, y_train, y_val = train_test_split(
    image_paths, mask_paths, test_size=0.3, random_state=42
)

# Dataset for train images
train_dataset = contrail.Dataset(
    x_train,
    y_train,
    augmentation=contrail.get_train_augmentation(),
    preprocessing=contrail.get_preprocessing(),
)

image, mask = train_dataset[0]
contrail.visualize(train_image=image, train_mask=mask)

# Dataset for validation images
val_dataset = contrail.Dataset(
    x_val,
    y_val,
    augmentation=contrail.get_val_augmentation(),
    preprocessing=contrail.get_preprocessing(),
)

image, mask = val_dataset[0]
contrail.visualize(val_image=image, val_mask=mask)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=16,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=16,
)

# %%

loss = "sr"

# Training detection model
model = contrail.ContrailModel(arch="UNet", in_channels=3, out_classes=1, loss=loss)

trainer = lightning.Trainer(max_steps=4000, log_every_n_steps=10)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# %%
torch.save(
    model.state_dict(), "data/model/contrail.torch.states.v[xx].[loss].[xxx].bin"
)
