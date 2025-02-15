# %%

import time

import data
import numpy as np
import torch
from contrail import ContrailModel
from torch.utils.data import DataLoader

torch.set_grad_enabled(False)


# %%
model1 = ContrailModel("UNet", in_channels=1, out_classes=1).cuda()
model1.load_state_dict(torch.load("data/models/own-dice-1000epoch.torch"))

model2 = ContrailModel("UNet", in_channels=1, out_classes=1).cuda()
model2.load_state_dict(torch.load("data/models/own-focal-1000epoch.torch"))

model3 = ContrailModel("UNet", in_channels=1, out_classes=1).cuda()
# model3.load_state_dict(torch.load("data/models/own-sr:dice-1000epoch.torch"))
model3.load_state_dict(torch.load("data/models/google:fewshot:400-sr-90minute.torch"))


# %%
train_dataset, test_dataset = data.own_dataset(for_training=False)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=16,
    shuffle=True,
)

batch = next(iter(test_dataloader))
batch = (batch[0].cuda(), batch[1].cuda())  # Move batch to GPU

model1.eval()
logits1 = model1(batch[0])
pred1 = logits1.sigmoid().cpu()  # back to cpu

model2.eval()
logits2 = model2(batch[0])
pred2 = logits2.sigmoid().cpu()

model3.eval()
logits3 = model3(batch[0])
pred3 = logits3.sigmoid().cpu()


for i in range(len(batch[0])):
    image = batch[0][i].cpu()
    labeled = batch[1][i].cpu()

    d = {
        "Image": np.array(image),
        "Labeled": np.array(labeled),
        "Dice": np.array(pred1[i]),
        "Focal": np.array(pred2[i]),
        "SR:Dice": np.array(pred3[i]),
    }

    data.visualize(**d)


# %%

# train_dataset, test_dataset = data.own_dataset(for_training=False)
train_dataset, test_dataset = data.google_dataset(
    for_training=False, contrail_only=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=16,
    shuffle=False,
)

batch = next(iter(test_dataloader))
batch = (batch[0].cuda(), batch[1].cuda())  # Move batch to GPU


model_names = ["Dice", "Focal", "SR"]

with torch.no_grad():
    for i, model in enumerate([model1, model2, model3]):
        model.eval()

        # flops = torchprofile.profile_macs(model, batch[0])

        torch.cuda.synchronize()
        start_time = time.time()
        logits = model(batch[0])
        torch.cuda.synchronize()
        end_time = time.time()

        logits = model(batch[0])
        predict = logits.sigmoid()
        target = batch[1]

        intersection = torch.nansum(predict * target, dim=(1, 2, 3))
        cardinality = torch.nansum(predict + target, dim=(1, 2, 3))
        dice_per_image = (2 * intersection) / cardinality
        dice = torch.mean(dice_per_image)

        total_time = end_time - start_time
        inference_time_per_image = total_time / batch[0].size(0)

        print(
            model_names[i],
            "\t",
            round(dice.item(), 2),
            "\tTime/img:",
            round(inference_time_per_image, 3),
        )

# %%
import matplotlib.pyplot as plt

results = []

for n in [30, 45, 50]:
    print(n)

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(batch[0][n].cpu().squeeze())

    plt.subplot(132)
    plt.imshow(target[n].cpu().squeeze())

    plt.subplot(133)
    plt.imshow(predict[n].cpu().squeeze())
    plt.show()

    results.append(
        {
            "Image": batch[0][n].cpu().squeeze().numpy(),
            "Labeled": target[n].cpu().squeeze().numpy(),
            "Predicted": predict[n].cpu().squeeze().numpy(),
        }
    )

# %%
import pickle

pickle.dump(results, open("failed_cases.pkl", "wb"))
