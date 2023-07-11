# %%
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm

import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

import albumentations as albu

sm.set_framework("tf.keras")


#%%

image_paths = sorted(glob.glob(f"data/unet/training_data_*/images/*.tiff"))
mask_paths = sorted(glob.glob(f"data/unet/training_data_*/masks/*.tiff"))

image_width = 32 * 7
image_height = 32 * 7

#%%

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(10, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


# classes for data loading and preprocessing
class Dataset:
    """Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

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

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.image_paths[i], cv2.IMREAD_COLOR)
        # image = cv2.resize(image, (image_height, image_width))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.astype(float) / 255

        if self.has_mask:
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
            # mask = cv2.resize(mask, (image_height, image_width))
            # mask = mask.astype(float)

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]

            return image, mask

        else:
            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image)
                image = sample["image"]

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample["image"]

        return image

    def __len__(self):
        return len(self.ids)


class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# Lets look at data we have
dataset = Dataset(image_paths, mask_paths)

image, mask = dataset[0]  # get some sample
visualize(image=image, mask=mask)


#%%

# define augmentations


def get_train_augmentation():
    transform = [
        # albu.HorizontalFlip(p=0.5),
        # albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=180,
            shift_limit=0.3,
            border_mode=0,
            p=1,
        ),
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.Resize(320, 320),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(transform)


def get_val_augmentation():
    transform = [
        # albu.HorizontalFlip(p=0.5),
        # albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=180,
            shift_limit=0.3,
            border_mode=0,
            p=1,
        ),
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.Resize(320, 320),
    ]
    return albu.Compose(transform)


def get_test_augmentation():
    transform = [
        albu.PadIfNeeded(
            min_height=320, min_width=320, always_apply=True, border_mode=0
        ),
        albu.Resize(320, 320),
    ]
    return albu.Compose(transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
    ]
    return albu.Compose(_transform)


dataset = Dataset(image_paths, mask_paths, augmentation=get_train_augmentation())
image, mask = dataset[0]
visualize(image=image, mask=mask)

dataset = Dataset(image_paths, mask_paths, augmentation=get_val_augmentation())
image, mask = dataset[0]
visualize(image=image, mask=mask)


# %%
backbone = "resnet34"

sm_preprocessing = sm.get_preprocessing(backbone)

model = sm.Unet(
    backbone,
    classes=1,
    encoder_weights="imagenet",
    # input_shape=(image_width, image_height, 3),
)

# inp = tf.keras.layers.Input(shape=(None, None, 1))
# l1 = tf.keras.layers.Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
# out = base_model(l1)

# model = tf.keras.Model(inp, out, name=base_model.name)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True,
        alpha=0.9,
        gamma=0,
        from_logits=True,
    ),
    metrics=["accuracy"],
)

#%%
batch_size = 8
epochs = 500

retrain = True

if retrain:

    x_train, x_test, y_train, y_test = train_test_split(
        image_paths, mask_paths, test_size=0.4, random_state=42
    )

    # Dataset for train images
    train_dataset = Dataset(
        x_train,
        y_train,
        augmentation=get_train_augmentation(),
        preprocessing=get_preprocessing(sm_preprocessing),
    )

    # Dataset for validation images
    val_dataset = Dataset(
        x_test,
        y_test,
        augmentation=get_val_augmentation(),
        preprocessing=get_preprocessing(sm_preprocessing),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = Dataloder(val_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    # print(train_dataloader[0][0].shape)
    # print(train_dataloader[0][1].shape)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "./best_model.h5", save_weights_only=True, save_best_only=True, mode="min"
        ),
        # tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50),
    ]

    # train model
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_dataloader,
        validation_steps=len(val_dataloader),
    )

#%%
model.load_weights("best_model.h5")


#%%

test_image_paths = glob.glob("data/unet/training_data_*/testing_images/*.tiff")
test_dataset = Dataset(
    test_image_paths,
    augmentation=get_test_augmentation(),
    preprocessing=get_preprocessing(sm_preprocessing),
)

for image in test_dataset:
    # image = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)

    visualize(
        image=image.squeeze(),
        pred_mask=pred_mask.squeeze(),
    )

#%%
for image, true_mask in val_dataset:
    image = np.expand_dims(image, axis=0)
    pred_mask = model.predict(image)

    visualize(
        image=image.squeeze(),
        true_mask=true_mask.squeeze(),
        pred_mask=pred_mask.squeeze(),
    )

# %%
