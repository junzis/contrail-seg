# Contrail detection and segmentation with neural networks


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

[Optional] If you want to train the model, run `train.py` script.

All trained model data are stored under `model` directory.


### Step 4: Visualize contrail detection

`detect.ipynb` provide examples for loading models and examples for detecting contrails.


## TensorFlow

There is also a sample code developed for TensorFlow users. The following dependencies have to be installed:


```
conda activate contrail
mamba install tensorflow scikit-learn matplotlib
pip install segmentation-models albumentations
```

The script `tensorflow_model/contrail_tf_keras.py` show the basic implementation of the segmentation model.
