# Neural network models for contrail detection and segmentation

Introducing an open-source project that implements contrail segmentation neural network models in PyTorch. 

These ResUNet models are constructed using fewshot augmented transfer learning, wherein multiple image augmentations are applied to a pre-trained ResUNet model. Consequently, the model can be efficiently fine-tuned using only a small set of labeled satellite images. 

To enhance contrail detection through contrail information in Hough space, **a new loss function, SR Loss**, has been developed, aiming at further optimizing the contrail detection process.

## Contrail detection examples

The following set of images shows examples of contrails detected with different models trained with Dice, Focal, and SR loss functions:

![contrail-detect-1](./static/contrail-detect-1.png)


The following set of images shows examples of contrails detected from different image sources, which are different from the GOES-16 images that are used in the model training.

![contrail-detect-2](./static/contrail-detect-2.png)



## Dependencies

* PyTorch
* segmentation_models_pytorch
* opencv
* scikit-learn


## Setup

It is recommended to use a mamba/conda environment. To setup mamba/conda, follow this tutorial: https://youtu.be/Ket0WUTm5JU?t=47

### Create conda environment

```
mamba create -n contrail python=3.11 -c conda-forge
```

### Install dependencies


For PyTorch users with CUDA 12.1 (recommended):

```bash
conda activate contrail
mamba install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install segmentation-models-pytorch albumentations
```

For PyTorch users without CUDA:

```bash
conda activate contrail
conda install pytorch cpuonly -c pytorch
pip install segmentation-models-pytorch albumentations
```


## Models

### Use pre-trained models

You can also download the already trained model weights from: https://surfdrive.surf.nl/files/index.php/s/n1b0L2qfu2PZ6d3

Save the downloaded models in a folder called `models` under the `data` directory.

### Train models with own data

If you want to train the model, you can train models with following examples:

```bash
# train with own dataset, for 1000 epoch, SR loss function
python train.py --dataset own --epoch 1000 --loss sr

python train.py --dataset own --epoch 1000 --loss dice

# train model for 60 minutes
python train.py --dataset own --time 60 --loss focal
```

### Train models with Google contrail dataset

You can also use Google contrail dataset for training the models

1. Download the data from: https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/data

2. Then put the uncompressed files in `google-goes-contrail` folder under `data` directory

3. Run `process_google_data.py` to generate mask statistics file `mask_stats.csv` inside of the previous folder

Then, use the following examples to train models

```bash
# train with google dataset, 30 minutes, SR loss
python train.py --dataset google --time 30 --loss sr

# fewshot training, 400 images, 30 minutes
python train.py --dataset google:fewshot:400 --time 30 --loss dice
```


## Detection and visulization

The `detect.py` provides examples for loading models and detecting contrails in the testing images.


