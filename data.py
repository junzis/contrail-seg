# %%
import glob

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_IMG_SIZE = 320

GOOGLE_DIR = "data/google-goes-contrail"


# %%
# visualization a series of images
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(n * 5, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image.squeeze())
    plt.tight_layout()
    plt.show()


def get_train_augmentation(image_size=DEFAULT_IMG_SIZE):
    """Define augmentation for contrail training images."""

    transform = [
        albu.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=180,
            shift_limit=0.3,
            border_mode=0,
            p=1,
        ),
        albu.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            always_apply=True,
            border_mode=0,
        ),
        albu.Resize(image_size, image_size),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
    ]
    return albu.Compose(transform)


def get_val_augmentation(image_size=DEFAULT_IMG_SIZE):
    """Define augmentation for contrail validation images."""

    transform = [
        albu.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=180,
            shift_limit=0.3,
            border_mode=0,
            p=1,
        ),
        albu.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            always_apply=True,
            border_mode=0,
        ),
        albu.Resize(image_size, image_size),
    ]
    return albu.Compose(transform)


def get_test_augmentation(image_size=DEFAULT_IMG_SIZE):
    """Define augmentation for contrail testing images (pad and resize only)."""
    transform = [
        albu.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            always_apply=True,
            border_mode=0,
        ),
        albu.Resize(image_size, image_size),
    ]
    return albu.Compose(transform)


def get_preprocessing():
    """Construct preprocessing transform

    Return:
        transform: albumentations.Compose

    """

    def to_tensor(input, **kwargs):
        return np.expand_dims(input, 0).astype("float32")

    _transform = [
        # albu.Lambda(image=smp.encoders.get_preprocessing_fn(ENCODER_NAME)),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class BaseDataset:
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder, Default: None
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, perspective, gamma, etc.)
        preprocessing (albumentations.Compose): data preprocessing from pre-trained model
            (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(
        self, image_paths, mask_paths=None, augmentation=None, preprocessing=None
    ):
        self.ids = image_paths
        self.image_paths = image_paths

        self.has_mask = True if mask_paths is not None else False

        if self.has_mask:
            self.mask_paths = mask_paths

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i): ...

    def __len__(self):
        return len(self.ids)


class OwnDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder, Default: None
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, perspective, gamma, etc.)
        preprocessing (albumentations.Compose): data preprocessing from pre-trained model
            (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        # read image in color
        image = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE)

        if self.has_mask:
            # read mask (png) and convert to grayscale
            mask = np.amax(cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED), axis=2)
            mask = mask / mask.max()

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

        else:
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample["image"]

            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample["image"]

        # normalize image
        image = (image - image.min()) / (image.max() - image.min())

        if self.has_mask:
            return image, mask
        else:
            return image


class GoogleDataset(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder, Default: None
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, perspective, gamma, etc.)
        preprocessing (albumentations.Compose): data preprocessing from pre-trained model
            (e.g. normalization, shape manipulation, etc.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ids = ["/".join(p.split("/")[-2:]) for p in self.image_paths]

    def __getitem__(self, i):
        band13 = np.load(f"{self.image_paths[i]}/band_13.npy")
        band15 = np.load(f"{self.image_paths[i]}/band_15.npy")

        image = (band13 - band15)[..., 4]

        if self.has_mask:
            mask = np.load(f"{self.mask_paths[i]}/human_pixel_masks.npy").squeeze()

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

        else:
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample["image"]

            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample["image"]

        # normalize image
        image = (image - image.min()) / (image.max() - image.min())

        if self.has_mask:
            return image, mask
        else:
            return image


def own_dataset(train=True):
    image_paths = sorted(glob.glob(f"data/goes/**/image/*.png"))
    mask_paths = sorted(glob.glob(f"data/goes/**/mask/*.png"))

    x_train, x_val, y_train, y_val = train_test_split(
        image_paths, mask_paths, test_size=0.3, random_state=42
    )

    train_dataset = OwnDataset(
        x_train,
        y_train,
        augmentation=get_train_augmentation() if train else get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )

    val_dataset = OwnDataset(
        x_val,
        y_val,
        augmentation=get_val_augmentation() if train else get_test_augmentation(),
        preprocessing=get_preprocessing(),
    )

    return train_dataset, val_dataset


def google_dataset(for_training=True, contrail_only=False, threshold=100):
    mask_stats = pd.read_csv(f"{GOOGLE_DIR}/mask_stats.csv")

    if for_training:
        # balance the data
        selected_records = pd.concat(
            [
                mask_stats.query("mask_pixels>200"),
                mask_stats.query("mask_pixels<200").sample(2000, random_state=42),
            ]
        )
    else:
        selected_records = mask_stats

    if contrail_only:
        selected_records = selected_records.query(f"mask_pixels>{threshold}")

    train_records = selected_records.query("tag=='train'").record_id.tolist()

    val_records = selected_records.query("tag=='validation'").record_id.tolist()

    x_train = y_train = [
        p
        for p in sorted(glob.glob(f"{GOOGLE_DIR}/*/*"))
        if "/".join(p.split("/")[-2:]) in train_records
    ]

    x_val = y_val = [
        p
        for p in sorted(glob.glob(f"{GOOGLE_DIR}/*/*"))
        if "/".join(p.split("/")[-2:]) in val_records
    ]

    train_dataset = GoogleDataset(
        x_train,
        y_train,
        augmentation=(
            get_train_augmentation() if for_training else get_test_augmentation()
        ),
        preprocessing=get_preprocessing(),
    )

    val_dataset = GoogleDataset(
        x_val,
        y_val,
        augmentation=(
            get_val_augmentation() if for_training else get_test_augmentation()
        ),
        preprocessing=get_preprocessing(),
    )

    return train_dataset, val_dataset


# %%
def google_dataset_few_shot(for_training=True, n=50):
    mask_stats = pd.read_csv(f"{GOOGLE_DIR}/mask_stats.csv")

    df_train = mask_stats.query("tag=='train'")
    train_samples = pd.concat(
        [
            df_train.query("0<mask_pixels<1000").sample(int(n * 0.3), random_state=42),
            df_train.query("1000<mask_pixels").sample(int(n * 0.7), random_state=42),
        ]
    )

    df_val = mask_stats.query("tag=='validation'")
    val_samples = pd.concat(
        [
            df_val.query("mask_pixels==0").sample(50, random_state=42),
            df_val.query("mask_pixels>0").sample(200, random_state=42),
        ]
    )

    train_records = train_samples.record_id.tolist()
    val_records = val_samples.record_id.tolist()

    x_train = y_train = [
        p
        for p in sorted(glob.glob(f"{GOOGLE_DIR}/*/*"))
        if "/".join(p.split("/")[-2:]) in train_records
    ]

    x_val = y_val = [
        p
        for p in sorted(glob.glob(f"{GOOGLE_DIR}/*/*"))
        if "/".join(p.split("/")[-2:]) in val_records
    ]

    train_dataset = GoogleDataset(
        x_train,
        y_train,
        augmentation=(
            get_train_augmentation() if for_training else get_test_augmentation()
        ),
        preprocessing=get_preprocessing(),
    )

    val_dataset = GoogleDataset(
        x_val,
        y_val,
        augmentation=(
            get_train_augmentation() if for_training else get_test_augmentation()
        ),
        preprocessing=get_preprocessing(),
    )

    return train_dataset, val_dataset
