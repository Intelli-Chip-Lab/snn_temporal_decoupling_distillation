# Efficient Distillation of Deep Spiking Neural Networks for Full-Range Timestep
Deployment

Efficient Distillation of Deep Spiking Neural Networks for Full-Range Timestep
Deployment

---

**Table of contents:**

- [Abstract](#abstract)
- [Dependency](#dependency)
- [Directory Tree](#directory)
- [Usage](#usage)

## Abstract

Spiking Neural Networks (SNNs) are emerging as a brain-
inspired alternative to traditional Artificial Neural Net-
works (ANNs), prized for their potential energy efficiency
on neuromorphic hardware. Despite this, SNNs often suffer
from accuracy degradation compared to ANNs and face de-
ployment challenges due to fixed inference timesteps, which
require retraining for adjustments, limiting operational flex-
ibility. To address these issues, our work considers the
spatio-temporal property inherent in SNNs, and proposes
a novel distillation framework for deep SNNs that optimizes
performance across full-range timesteps without specific re-
training, enhancing both efficacy and deployment adapt-
ability. We provide both theoretical analysis and empirical
validations to illustrate that training guarantees the con-
vergence of all implicit models across full-range timesteps.
Experimental results on CIFAR-10, CIFAR-100, CIFAR10-
DVS, and ImageNet demonstrate state-of-the-art perfor-
mance among distillation-based SNNs training methods.
<img src="doc/figure/fig.png" alt="introduction_figure" style="zoom:100%;" />


## Dependency

The major dependencies of this repo are listed as below.

```
# Name                 Version
python                  3.10.14 
torch                   2.4.1
torchvision             0.19.1
tensorboard             2.17.1
spikingjelly            0.0.0.0.14
```

## Directory Tree

```
├── experiment
│   ├── cifar
│   │   ├── __init__.py
│   │   ├── ann.py
│   │   ├── config
│   │   └── main.py
│   └── dvs
│       ├── __init__.py
│       ├── config
│       ├── main.py
│       └── process.py
├── model
├── util
│   ├── __init__.py
│   ├── data.py
│   ├── image_augment.py
│   ├── misc.py
│   └── util.py




```

The experiment code for static image are located on directory(`experiment/cifar/main.py`).
The experiment code for event-stream dataset are located on directory(`experiment/dvs/main.py`).
The code associated with neurons is defined in the file(`model/layer.py`).

## Usage

1. Try to reproduce the results on the CIFAR-10 dataset with the following command:
    ```bash
    python experiment/cifar/main.py --stu_arch resnet18 --tea_arch resnet34 --dataset CIFAR10 --train_batch_size 128 --val_batch_size 128 --data_path [data_path] --tea_path [your pt file]  --wd 5e-4 --decay 0.5 --T 6 --num_epoch 300 --alpha 0.2 --beta 0.5 --lr 0.1
    ```
   
2. Try to train on the CIFAR-10 dataset with the following command:
    ```bash
    python experiment/cifar/ann.py --arch resnet18  --dataset CIFAR10 --train_batch_size 128 --val_batch_size 128 --data_path [data_path]  --wd 5e-4 --epoch 300 lr 0.1 
    ```

3. Try to reproduce the results on the CIFAR-100 dataset with the following command:
    ```bash
    python experiment/cifar/main.py --stu_arch resnet18 --tea_arch resnet34 --dataset CIFAR10 --train_batch_size 128 --val_batch_size 128 --data_path [data_path] --tea_path [your pt file] --wd 5e-4 --decay 0.5 --T 6 --num_epoch 300 --alpha 0.2 --beta 0.5 --lr 0.1
    ```

4. Try to reproduce the results on the DVSCIFAR-10 dataset with the following command:
    ```bash
    python experiment/dvs/main.py --stu_arch resnet19 --tea_arch resnet19 --dataset CIFAR10_DVS_Aug --train_batch_size 32 --val_batch_size 32 --data_path [data_path] --tea_path [your pt file] --wd 5e-4 --decay 0.5 --T 6 --num_epoch 300  --alpha 0.2 --beta 0.5 --lr 0.1
    ```
   
5. Try to reproduce the results on the ImageNet dataset with the following command:
    ```bash
    torchrun --nproc_per_node=8 experiment/imagenet/main.py --stu_arch preact_resnet34 --tea_arch resnet34 --dataset imagenet --train_batch_size 512 --val_batch_size 512 --data_path [data_path] --tea_path [your pt file] --wd 2e-5 --decay 0.2 --T 4 --num_epoch 100  --alpha 0.2 --beta 0.5 --lr 0.2
    ```
