# DSGAN-Tensorflow
Unofficial Tensorflow Implementation of [Diversity-Sensitive Conditional Generative Adversarial Networks](https://openreview.net/forum?id=rJliMh09F7). 

Official PyTorch implementation and project page can be found [here](https://github.com/maga33/DSGAN).

I have only implemented Inpainting application, though other applications can be included with trivial changes to the network architectures! This repository will be updated in the near future for the other applications.

## Inpainting

Below figure, illustrates inpainting for 8 randomly selected images from validation set with center masking. In each row, different noise code is used to show effectiveness of the method. With diversity loss, different noise codes correspond to different semantically meaningful images. 
![Inpainting illustration](https://github.com/yasinyazici/DSGAN-Tensorflow/blob/master/examples/inpainting.jpg)

