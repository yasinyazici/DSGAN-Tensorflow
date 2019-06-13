# DSGAN-Tensorflow
Unofficial Tensorflow Implementation of [Diversity-Sensitive Conditional Generative Adversarial Networks](https://openreview.net/forum?id=rJliMh09F7). 

Official PyTorch implementation and project page can be found [here](https://github.com/maga33/DSGAN).

I have only implemented Inpainting application, though other applications can be included with trivial changes to the network architectures! This repository will be updated in the near future for the other applications.

## Inpainting

Below figure, illustrates inpainting for 8 randomly selected images from validation set with center masking. In each row, different noise code is used to show effectiveness of the method. With diversity loss, different noise codes correspond to different semantically meaningful images. 
![Inpainting illustration](https://github.com/yasinyazici/DSGAN-Tensorflow/blob/master/examples/inpainting.jpg)

### Setup
- Download the [CelebA dataset](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view).
    
- Change `data_dir` in `main.py` to wherever your 'img_align_celeba' file is.

### Training
Train the model with following script

`python train.py`

The training generates illustration like above at every 500 iteration.


