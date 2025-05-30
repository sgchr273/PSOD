This is the official repository for the paper "Class-Specific Subspace Alignment for Effective
Out-of-Distribution Detection".

1. **Data arrangement**<br>
We downloaded and organized the three in-distribution datasets used in our experiments. Both CIFAR-10 and ImageNet include two subfolders—id_data and ood_data—where:
id_data contains the in-distribution files for that dataset while ood_data holds the out-of-distribution samples. The hierarchical structure of all three datasets is shown below:

- **CIFAR10**
  - **id_data**
    - cifar-10-batches-py
    - cifar-10-python.tar.gz
  - **ood_data**
    - iSUN
    - LSUN
    - SVHN
- **CIFAR100**
  - cifar-100-python
  - cifar-100-python.tar.gz
- **Imagenet**
  - **id_data**
    - n01440764
    - n01443537
    - n01484850
    - n01491361 <br>
          . <br>
          . <br>
          . <br>
  - **ood_data**
    - iNaturalist
    - Places
    - SUN
    - Textures
2. **Saved Model Checkpoints**<br>
Out of four neural network architectures used int he experiments, the checkpoints for the resnet18 pretrained on cifar10 and cifar100 and checkpoint of resnet34 for cifar10 and cifar100 can be downloaded from here:




Some of the code has been borrowed from the repsitory of the paper "NECO: NEURAL COLLAPSE BASED OUT-OFDISTRIBUTION DETECTION".
