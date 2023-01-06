# Neural Proxies for Darktable

<br/>

## Getting Started

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