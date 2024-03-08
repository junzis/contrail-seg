# %%
import glob
import warnings

import lightning
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import contrail

warnings.filterwarnings("ignore")

# %%

# fmt: off
train_ids = [3, 8, 9, 11, 22, 28, 30, 31, 34, 35, 36, 37, 38, 44, 53, 55, 67, 68, 72, 74, 75, 77, 79, 80, 87, 88, 89, 91, 97, 101, 105, 107, 108, 110] 
val_ids = [7, 12, 58, 60, 61, 69, 70, 78, 84, 85, 86, 90, 92, 93, 94, 95, 96, 98, 99, 100, 106]
# fmt: on

x_train = []
y_train = []
x_val = []
y_val = []

for folder in sorted(glob.glob("data/landsat/lowres/*")):
    input = glob.glob(f"{folder}/*_B9.png")
    output = glob.glob(f"{folder}/label/*_contrail.png")

    # input = glob.glob(f"{folder}/*_B7.png")
    # output = glob.glob(f"{folder}/label/*_shadow.png")

    if len(input) != 1 or len(output) != 1:
        continue

    if int(folder.split("_")[1]) in train_ids:
        x_train.append(input[0])
        y_train.append(output[0])

    if int(folder.split("_")[1]) in val_ids:
        x_val.append(input[0])
        y_val.append(output[0])


# %%
# x_train, x_val, y_train, y_val = train_test_split(
#     image_paths, mask_paths, test_size=0.3, random_state=42
# )

# Dataset for train images
train_dataset = contrail.Dataset(
    x_train,
    y_train,
    augmentation=contrail.get_train_augmentation(),
    preprocessing=contrail.get_preprocessing(),
)

# Dataset for validation images
val_dataset = contrail.Dataset(
    x_val,
    y_val,
    augmentation=contrail.get_val_augmentation(),
    preprocessing=contrail.get_preprocessing(),
)

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
# for i, data in enumerate(train_dataset):
#     image, mask = data
#     print(i)
#     contrail.visualize(train_image=image, train_mask=mask)

# %%
# for data in val_dataset:
#     image, mask = data
#     contrail.visualize(train_image=image, train_mask=mask)


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

# trainer.fit(
#     model,
#     ckpt_path="lightning_logs/version_16/checkpoints/epoch=1999-step=6000.ckpt",
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )


# %%
torch.save(
    model.state_dict(),
    "data/models/contrail.torch.states.v205_contrail_landsat_4000.bin",
)

# %%

# model = contrail.ContrailModel(arch="UNet", in_channels=3, out_classes=1, loss=loss)
# model.load_state_dict(
#     torch.load("data/models/contrail.torch.states.v17_shadow_landsat.bin")
# )

# %%
# val_batch = next(iter(val_dataloader))

# with torch.no_grad():
#     model.eval()
#     logits = model(val_batch[0])
#     pred_masks_1 = logits.sigmoid()

# for i in range(10):
#     image = val_batch[0][i]
#     labeled = val_batch[1][i]
#     pred1 = pred_masks_1[i]

#     d = {
#         "Image": np.array(image),
#         "Labeled": np.array(labeled),
#         "Predict": np.array(pred1),
#     }

#     contrail.visualize(**d)
