# Neural networks models for contrail detection and segmentation


This is a repository contains code that implements contrail segmentation neural network models in PyTorch. The models are built using augmented transfer learning with a customized loss function, SR Loss, which optimizes the contrail detection using contrail information in Hough space.


## Contrail detection examples

The follow set of images shows the examples of contrails detected with different models trained with Dice, Focal, and SR loss functions:

![contrail-detect-1](./static/contrail-detect-1.png)


The follow set of images shows the examples of contrails detected from vary image sources, which are different from the GOES-16 images that are use in the model training. It demonstrates that the model performs well on other image sources.

![contrail-detect-2](./static/contrail-detect-2.png)



## Dependencies

* PyTorch
* segmentation_models_pytorch
* opencv
* scikit-learn


## Setup

You should already have a work mamba/conda environment. If not, follow this tutorial to set it up: https://youtu.be/Ket0WUTm5JU?t=47


### Step 1: Create a conda environment 

Note: replace `mamba` with `conda` for slower installation :)


```
mamba create -n contrail python=3.10 -c conda-forge
```

### Step 2: Install dependencies


* For PyTorch users with CUDA 11.8 (recommended):

    ```
    conda activate contrail
    mamba install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install segmentation-models-pytorch albumentations
    ```

* For PyTorch users without CUDA:

    ```
    conda activate contrail
    conda install pytorch cpuonly -c pytorch
    pip install segmentation-models-pytorch albumentations
    ```


### Step 3: Models

[Optional] If you want to train the model, take a look and run the `train.py` script.

You can also download the already trained model weights from: https://surfdrive.surf.nl/files/index.php/s/n1b0L2qfu2PZ6d3

Save the the models in a folder called `models` under the `data` directory.


### Step 4: Visualize contrail detection

`detect.ipynb` provide examples for loading models and examples for detecting contrails.


## What about TensorFlow?

The script `tensorflow_model/contrail_tf_keras.py` shows a basic implementation of the segmentation model in TensorFlow. Note this code is not fully tested.

To test your model in tensorflow, the following dependencies have to be installed:

```
conda activate contrail
mamba install tensorflow scikit-learn matplotlib
pip install segmentation-models albumentations
```
