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

### Build our Modified Version of Darktable
This repository still relies on Darktable's CLI to generate ground truth data for the neural proxies. Before running any code, make sure that you have __Darktable__ built on your system. The scripts in this repository are configured to run with a custom version of Darktable that saves intermediate tap-outs (inputs and outputs to each image processing operation) in between each block. Therefore, our version must be built in order to access all of this codebase's functionality.

### Set Darktable Path
The various parameters and constants used throughout this repository are all stored in ```Darktable_constants.py```. One of these __must__ be set to match your system specifications before running. In the file, replace ```DARKTABLE_PATH``` with the correct, absolute path to ```darktable-cli.exe``` on your own system

<br/>
After completing these two steps, the code should run without a hitch!

## A Tour of the Codebase

In order to make the most of this codebase, there are only a handful of scripts that need to be understood.

### Darktable_constants.py
This file contains all of the constants used throughout every other script in this repository. Many of these constants can be changed in order to customize the codebase's behavior and functionality. In other words, this can be thought of as user "settings."

There are several key variables in this file that are worth noting:
* ```INTERACTIVE```: This toggles whether or not scripts give interactive prompts between key parts of the training process
* ```GENERATE_STAGE_1 & GENERATE_STAGE_2``` : These two booleans toggle whether or not data is generated for proxy training (stage 1) or proxy evaluation (stage 2), respectively. If you already have data generated for one of these steps, its corresponding variable should be set to ```False``` to avoid generating again for no reason. Therefore, it is crucial to be aware of these parameters and set them correctly each subsequent time you run any code.
* ```GENERATE_WITH_CHECKPOINTS```: This toggles an alternate data generation method where checkpoints are saved along the way. This is particularly useful if the code is being run in uncertain conditions, and interruption is a distinct possibility. By default, however, this parameter is set to ```False```.
* ```CREATE_ANIMATION & PIPELINE_CREATE_ANIMATION```: These toggle whether or not scripts save model outputs for visualizing with __FFmpeg__ for single-model evaluation or pipeline-wide evaluation, respectively. By default, both of these are set to ```True```.

Many of the other constants in this file can prove quite useful for customizing your experience. It is recommended that you read through ```Darktable_constants.py``` yourself, as all parameters are carefully annotated there.

### Darktable_proxies.py
This is the all-in-one script to generate data, train individual neural proxies for Darktable blocks,  and evaluate the performance of those proxies (in that order).  This will be the first script that you run in this codebase.

### Darktable_eval.py 
This script is a quick way to evaluate any neural proxies on a specific, user-selected input. It handles both data generation and evaluation. It is useful if there is ever a need to check how a certain model is performing on a particular input.

### Darktable_sweep.py
This script is another great way to visualize the performance of individual neural proxies. It uniformly samples points across the entire range of input parameters for a given proxy, generating data for those points, and evaluating the model on them. The model outputs for these points can then be animated together with __FFmpeg__, yielding a helpful video that sweeps across the entire range of inputs for a proxy, showing how outputs gradually change.

### Pipeline_regression.py
This is the natural next step after running ```Darktable_proxies.py``` for all of Darktable's blocks and having a set of functioning proxies. This script chains together individual neural proxies into a pipeline, generates data, and evaluates the performance of the pipeline.

### Various Utils

## How to Run the Code
To assemble a functioning clone of Darktable, only two scripts need to be run.

### Train Proxies with Darktable_proxies.py
For each Darktable image processing block that needs to be learned, run the ```Darktable_proxies``` script. This will take care of all the data generation, training, and model evaluation/visualization.
<br/>

How to use the script:
```
python Darktable_proxies.py [proxy_type]_[parameter] [# of points to sample for training data]
```
  *__proxy_type__ is the name of the Darktable block that is being learned, while __paramater__ is the name of the parameter in that block being trained. For optimal performance, it is recommended to use a larger number of points to sample for the training data (at least 500).*
<br/>
#### Proxy Trainng
Each time it runs, the script will create directories to store model weights from each training epoch, as well as model predictions (these can prove helpful to see how well the model is doing). These directories are named based on the proxy type and parameter being trained on:
- [proxy_type]_[parameter]_model_weights
- [proxy_type]_[parameter]_predictions

Ultimately, the only essential files are the model weights. By default, model predictions are saved every 50 training epochs, though this can be adjusted by changing ```SAVE_OUTPUT_FREQ``` in ```Darktable_constants.py```.
<br/>

####  Proxy Evaluation
After it has finished training a given proxy, this script will automatically evaluate it. It does so by attempting a simple regression task with the model, where the proxy must iteratively improve its best guess as to what input parameter value produces the given output image. The final parameter guess of the model, as well as model outputs from each iteration of the regression (used to make animations) are saved in newly-created directories:
- params
- [proxy_type]_[parameter]_animations

The number of iterations performed for each regression task can be adjusted by changing ```PARAM_TUNING_NUM_ITER``` in ```Darktable_constants.py```.
<br/>

__Note:__ For a complete list of all possible proxy types and parameters, see the table at the top of this document.

### Assemble the Pipeline with Pipeline_regression.py or Darktable_pipeline.py
  
## Evalutating Models 

### Evaluating Individual Proxies 

### Evaluating the Entire Pipeline

## Visualizing Results

This repository includes a number of helper scripts and built-in functionality to visualize model performance and parameter regression.

### Install FFmpeg
Many of the scripts in this repository rely on __FFmpeg__ to produce animations of model performance. It can be installed at https://ffmpeg.org/download.html.

### Visualizing Proxy Training

The ```Darktable_sweep.py``` script is a convenient way to visualize a given proxy's performance over its entire range of possible parameters. This script samples values across an entire parameter range, evaluating the proxy on each value (all for a single image), and saves all of the proxy outputs as numbered images. These can then be stitched together with __FFmpeg__ in order to get a video of how model performances changes across values.  

How to use the script:

```
python Darktable_sweep.py [proxy type]_[parameter] [path to .DNG image to use as input] [integer number of paramater values to sample]

```

Once the script has finished running, use the following FFmpeg command to create the video:

```
ffmpeg -r 25 -f image2 -s 1920x1080 -i [image name]_[proxy type]_[parameter]_sweep_%04d.png -vcodec libx264 -crf 15 -pix_fmt yuv420p [VIDEO NAME].mp4

```

__Note:__ To visualize model performance for specific paramter values, use the ```Darktable_eval.py``` helper script instead.
### Visualizing Slider Regression
### Visualizing Pipeline Regression
## Helpful Tips