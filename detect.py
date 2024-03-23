# %%

import numpy as np
import torch
from torch.utils.data import DataLoader

import data
from contrail import ContrailModel

torch.set_grad_enabled(False)


# %%
model1 = ContrailModel("UNet", in_channels=1, out_classes=1)
model1.load_state_dict(torch.load("data/models/own-dice-1000epoch.torch"))

model2 = ContrailModel("UNet", in_channels=1, out_classes=1)
model2.load_state_dict(torch.load("data/models/own-focal-1000epoch.torch"))

model3 = ContrailModel("UNet", in_channels=1, out_classes=1)
model3.load_state_dict(torch.load("data/models/own-sr:dice-1000epoch.torch"))


# %%
train_dataset, test_dataset = data.own_dataset(train=False)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=16,
    shuffle=True,
)

batch = next(iter(test_dataloader))

model1.eval()
logits1 = model1(batch[0])
pred1 = logits1.sigmoid()

model2.eval()
logits2 = model2(batch[0])
pred2 = logits2.sigmoid()

model3.eval()
logits3 = model3(batch[0])
pred3 = logits3.sigmoid()


for i in range(len(batch[0])):
    image = batch[0][i]
    labeled = batch[1][i]

    d = {
        "Image": np.array(image),
        "Labeled": np.array(labeled),
        "Dice": np.array(pred1[i]),
        "Focal": np.array(pred2[i]),
        "SR:Dice": np.array(pred3[i]),
    }

    data.visualize(**d)


# %%
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1000,
    num_workers=16,
    shuffle=True,
)

batch = next(iter(test_dataloader))

model_names = ["Google, Dice", "Google, SR", "Own Data, Dice", "Own Data, SR"]

with torch.no_grad():
    for i, model in enumerate([model1, model2, model3, model4]):
        model.eval()
        logits = model(batch[0])
        predict = logits.sigmoid()
        target = batch[1]
        intersection = torch.nansum(predict * target)
        cardinality = torch.nansum(predict + target)
        dice = (2 * intersection) / cardinality
        print(model_names[i], "\t", round(dice.item(), 2))
