# OOD Augmentation May Be at Odds with Open-Set Recognition

This repository contains the code for the paper "[OOD Augmentation May Be at Odds with Open-Set Recognition](https://arxiv.org/abs/2206.04242)". In this paper, we have shown that Maximum Softmax Probability (MSP), as the simplest baseline for Open-Set Recognition (OSR), applied on Vision Transformers (ViTs) as the base classifier that is trained with non-OOD augmentations can surprisingly outperform many recent OSR methods.


<p align="center">
  <img src="https://user-images.githubusercontent.com/72752732/190904647-30048ef7-db7d-4e3b-a682-6d35879e5de9.PNG" />
</p>

## OOD-ness

**What is OOD-ness?** We call the difference between distribution of augmented data and original data as OOD-ness. This difference causes OOD samples to be classified as the closed set. 

In `ood_ness.py`, we have implemented two methods that measures OOD-ness for different augmentations with different levels.

## Standard and contrastive training and evaluating

To evaluate our claims about the effect of OOD-ness and diversity on OSR performance, the following commands can be used to train a standard model and measure its OSR performance. The augmentations are defined at the beginnig of training code using `transform_train`.

Standard training:
```
python train_ood_standard.py --epochs 100 --data cifar10 --known-classes '0 1 2 4 5 9' --model-type 'vitb32' --pretraining imagenet
```

OSR performance of standard model:
```
python test_ood_standard.py --data cifar10 --known-classes '0 1 2 4 5 9' --model-type 'vitb32' --pretraining imagenet
```

The arguments can be used to set the data, known classes in the training, architecture of the model, and the data used for pre-training of the model. 

To evaluate the effect of contrastive training on OSR, the following commands can be used for contrastive training of a model and measuring its OSR performance.

Contrastive training:
```
python train_ood_scl.py --data cifar10 --known-classes '0 1 2 4 5 9' --model-type 'vitb32'
```

OSR performance of contrastive training:
```
python test_ood_scl.py --data cifar10 --known-classes '0 1 2 4 5 9' --model-type 'vitb32'
```

## Citation

If you find this useful for your research, please cite the following paper:
```
@article{azizmalayeri2022ood,
  title={OOD Augmentation May Be at Odds with Open-Set Recognition},
  author={Azizmalayeri, Mohammad and Rohban, Mohammad Hossein},
  journal={arXiv preprint arXiv:2206.04242},
  year={2022}
}
```
