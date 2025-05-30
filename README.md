This is the official repository for the paper "Class-Specific Subspace Alignment for Effective
Out-of-Distribution Detection". The steps to run the experiment are provided below:

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
2. **Model Checkpoints**<br>
Among the four neural network architectures evaluated, the pretrained checkpoints for ResNet-18 and ResNet-34 on CIFAR-10 and CIFAR-100 are available in the 'model_checkpoints' folder.
For ResNet-50 and MobileNet, we used the ImageNet-pretrained weights provided by Torchvision.
3. **Experiments Running**<br>
PSOD can be experimented with using the following command:<br>
python main.py --id_data <id_data_name> --ood_data <ood_data_name>  --model_name <model_name><br>
For example, one scenario is given below:<br>
python main.py --id_data cifar10 --ood_data SVHN  --model_name resnet18 <br><br>
To simulate the other baseline methods, run the file test_all_ood.py where the configuration for id_data, ood_data and model_name can be set the same way we did above for PSOD.   




Some of the code has been borrowed from the repsitory of the paper "NECO: NEURAL COLLAPSE BASED OUT-OFDISTRIBUTION DETECTION" https://arxiv.org/abs/2310.06823.
