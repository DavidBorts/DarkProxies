# Neural Proxies for Darktable
This repository contains all of the code necessary to train and evaluate neural network proxies for many of __Darktable__'s image processing blocks. Ultimately, these neural proxies can be chained together to get a functioning replica of Darktable.
<br/>
This codebase does so by following these steps:

__1.__ Generate training data using Darktable's CLI

__2.__ Train neural proxies for individual Darktable blocks

* Evaluate proxy performance before moving on

__3.__ Chain blocks together into a single pipeline

* Evaluate pipeline performance
  

## Getting Started

To use the code in this repository, there are several essential prerequisites.

### Install Darktable
Before running any code, make sure that you have __Darktable__ installed on your system. This can be done for free at https://www.darktable.org/install/. This repository relies on Darktable's CLI to generate ground truth data for the neural proxies. 

### Customize Script Parameters
The various parameters and constants used throughout this repository are all stored in ```Darktable_constants.py```. Two of the variables in this file __must__ be set to user specifications before running.
<br/>
__1.__ Replace ```DARKTABLE_PATH``` with the correct, absolute path to ```darktable-cli.exe``` on your own system

__2.__ Replace ```IMAGE_ROOT_DIR``` with your desired root directory from which all relative file paths will be constructed. The variable is set to ```.``` by default, which keeps all new files in the same directory as this code, but that need not be the case.

<br/>
After completing these two steps, the code should run without a hitch!

## A Tour of the Codebase

## How to Run the Code

## Evalutating Models

### Evaluating Individual Proxies

### Evaluating the Entire Pipeline

## Visualizing Results

This repository includes a number of helper scripts and built-in functionality to visualize model performance and parameter regression.

### Visualizing Proxy Training
The ```Darktable_sweep.py``` script is a convenient way to visualize a given proxy's performance over its entire range of possible parameters. This script samples values across an entire parameter range, evaluating the proxy on each value (all for a single image), and saves all of the proxy outputs as numbered images. These can then be stitched together with __ffmpeg__ in order to get a video of how model performances changes across values.

How to use the script:
```
python Darktable_sweep.py [proxy type]_[parameter] [path to .DNG image to use as input] [integer number of paramater values to sample]
```

Once the script has finished running, use the following ffmpeg command to create the video:
```
ffmpeg -r 25 -f image2 -s 1920x1080 -i [image name]_[proxy type]_[parameter]_sweep_%04d.png -vcodec libx264 -crf 15  -pix_fmt yuv420p [VIDEO NAME].mp4
```

__Note:__ To visualize model performance for specific paramter values, use the ```Darktable_eval.py``` helper script instead.

### Visualizing Slider Regression

### Visualizing Pipeline Regression

## Helpful Tips