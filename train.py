# %%
import warnings

import click
import lightning
import torch
from torch.utils.data import DataLoader

import data
from contrail import ContrailModel

warnings.filterwarnings("ignore")


# %%


@click.command()
@click.option("--dataset", required=True)
@click.option("--minute", required=False, type=int, help="minutes")
@click.option("--epoch", required=False, type=int, help="minutes")
@click.option("--loss", required=True, help="dice, focal, or sr")
@click.option("--base", required=False, help="dice or focal (for sr loss)")
def main(dataset, minute, epoch, loss, base):

    print(
        f"training: {dataset} data, {minute} minutes, {epoch} epoch, {loss} loss, {base} base"
    )

    torch.cuda.empty_cache()

    if dataset == "own":
        train_dataset, val_dataset = data.own_dataset()
    elif dataset == "google":
        train_dataset, val_dataset = data.google_dataset()
    elif dataset.startswith("google:fewshot:"):
        n = int(dataset.split(":")[-1])
        train_dataset, val_dataset = data.google_dataset_few_shot(n=n)
    else:
        print(f"dataset: {dataset} unknown")
        return

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )

    model = ContrailModel(arch="UNet", in_channels=1, out_classes=1, loss=loss)

    # callback_save_model = lightning.callbacks.ModelCheckpoint(
    #     dirpath="data/models/",
    #     filename="google-dice-{epoch:02d}epoch.torch",
    #     save_top_k=-1,
    #     every_n_epochs=10,
    # )

    if minute is not None:
        trainer = lightning.Trainer(
            max_time=f"00:{(minute//60):02d}:{(minute%60):02d}:00",
            log_every_n_steps=20,
        )
        max_val = minute
        tag = "minute"

    elif epoch is not None:
        trainer = lightning.Trainer(
            max_epochs=epoch,
            log_every_n_steps=20,
        )
        max_val = epoch
        tag = "epoch"

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    if base is None:
        f_out = f"data/models/{dataset}-{loss}-{max_val}{tag}.torch"
    else:
        f_out = f"data/models/{dataset}-{loss}:{base}-{max_val}{tag}.torch"

    torch.save(model.state_dict(), f_out)


if __name__ == "__main__":
    main()
