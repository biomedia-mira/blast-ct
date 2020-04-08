# BLAST-CT
**B**rain **L**esion **A**nalysis and **S**egmentation **T**ool for **C**omputed **T**omography

This repository provides the source code of our traumatic brain injury image segmentation tool for 3D computed tomography scans. Pre-trained models will be available soon together with detailed instructions on how to use the code on new data.

An article describing our method and results is currently under review.

## Source code

The provided source code enables training and testing our convolutional neural network designed for multi-class brain lesion segmentation in CT.

## Pre-trained model

We also make available a model that has been trained on a set of 184 annotated CT scans obtained from multiple clinical sites. This model has been validated on a set of 655 CT scans. Additionally, our model has been validated on an external, independent set of scans from 500 patients from the CQ500 dataset.

## Installation

### Linux and MacOS
On a fresh python3 virtual environment

`pip install git+https://github.com/biomedia-mira/blast-ct.git`

### Windows

# Usage with examples

If you wish to follow along with examples please run the following in your bash console:
```
mkdir blast-ct-example
cd blast-ct-example
svn checkout "https://github.com/biomedia-mira/blast-ct/trunk/blast_ct/data/"
```

## Inference on one image
To run inference on one image using our pre-trained models:

`blast-ct --input <path-to-input-image> --output <path-to-output-image> --device <device-id>`

1. `--input`: path to the input input image which must be in nifti format (`.nii` or `.nii.gz`);
2. `--output`: path where prediction will be saved (with extension `.nii.gz`);
3. `--device <device-id>` the device used for computation. Can be `'cpu'` (up to 1 hour per image) or an integer 
indexing a cuda capable GPU on your machine. Defaults to CPU;
4. pass `--ensemble true`: to use an ensemble of 12 models which improves segmentation quality but slows down inference
 (recommended for gpu).

##### Working example:

In the `blast-ct-example` directory run (might take up to an hour on CPU):

`blast-ct --input data/scans/scan_0_image.nii.gz --output scan_0_prediction.nii.gz`


## Inference on multiple images
To run inference on multiple images using our ensemble of pre-trained models:

```
blast-ct-inference \
    --job-dir <path-to-job-dir> \
    --test-csv-path <path-to-test-csv> \ 
    --device <device-id>
```

1. `--job-dir`: the path to the directory where the predictions and logs will be saved;
2. `--test-csv-path`: the path to a [csv file](#csv-files-for-inference-and-training) containing the paths of the 
images to be processed;
3. `--device <device-id>` the device used for computation. Can be `'cpu'` (up to 1 hour per image) or an integer 
indexing a cuda capable GPU on your machine. Defaults to CPU;
4. pass `--overwrite true`: to write over existing `job-dir`.

##### Working example:

In the `blast-ct-example` directory run (GPU example):

`blast-ct-inference --job-dir my-inference-job --test-csv-path data/data.csv --device 0`


## Training models on your own data

To train your own model:

```
blast-ct-train \
    --job-dir <path-to-job-dir> \
    --config-file <path-to-config-file> \
    --train-csv-path <path-to-train-csv> \
    --valid-csv-path <path-to-valid-csv> \
    --num-epochs <num-epochs> \
    --device <gpu_id> \
    --random-seed <list-of-random-seeds>
```

1. `--job-dir`: the path to the directory where the predictions and logs will be saved;
2. `--config-file`: the path to a json config file (see `data/config.json` for example);
3. `--train-csv-path`: the path to a [csv file](#csv-files-for-inference-and-training) containing the paths of the 
images, targets and sampling masks used to train th model;
4. `--valid-csv-path`: the path to a [csv file](#csv-files-for-inference-and-training) containing the paths of the 
images used to keep track of the model's performance during training;
5. `--num-epochs`: the number of epochs for which to train the model (1200 was used with the example config)
6. `--device <device-id>` the device used for computation (`'cpu'` or integer indexing GPU). GPU is strongly recommended.
7. `-random-seeds`: a list of random seeds used for training. 
Pass more than one to train multiple models one after the other.
8. pass `--overwrite true`: to write over existing `job-dir`.


##### Working example:

In the `blast-ct-example` directory run:
```
blast-ct-train \
    --job-dir my-custom-inference-job \
    --config-file data/config.json \
    --train-csv-path data/data.csv \
    --valid-csv-path data/data.csv \
    --num-epochs 10 \
    --device 0 \
    --random-seeds "1"
```


## Inference with your model

To run inference with your own models and config use
```
blast-ct-inference \
   --job-dir <path-to-job-dir> \
   --config-file <path-to-config-file> \
   --test-csv-path <path-to-test-csv> \
   --device <gpu_id> \
   --saved-model-paths <list-of-paths-to-saved-models>
```

1. `--job-dir`: the path to the directory where the predictions and logs will be saved;
2. `--config-file`: the path to a json config file (see `data/config.json` for example);
4. `--test-csv-path`: the path to a [csv file](#csv-files-for-inference-and-training) containing the paths of the 
images to be processed;
4. `--device <device-id>` the device used for computation. Can be `'cpu'` (up to 1 hour per image) or an integer 
indexing a cuda capable GPU on your machine. Defaults to CPU;
 `--saved-model-paths` is a list of pre-trained model paths;
5. pass `--overwrite true`: to write over existing `job-dir`.

##### Working example:

In the `blast-ct-example` directory run (GPU example):

`blast-ct-inference --job-dir my-custom-inference-job --config-file data/config.json --test-csv-path data/data.csv 
--device 0 --saved-model-paths "data/saved_models/model_1.pt data/saved_models/model_2.pt data/saved_models/model_6.pt"`



## csv files for inference and training

The tool takes input from csv files containing lists of images with unique ids.
Each row in the csv represents a scan and must contain:
1. A column named `id` which must be unique for each row (otherwise overwriting will happen);
2. A column named `image` which must contain the path to a nifti file;
3. (training only) A column named `target` containing a nifti file with the corresponding labels for training;
4. (training only; optional) A column named `sampling_mask` containing a nifti file with the corresponding sampling mask 
for training;
See `data/data.csv` for a working example (with only one row).
