import argparse

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from numpy.linalg import pinv
from scipy.special import softmax
import pickle
# import extract_utils
import ood_methods
from multiple_subspaces import ResNet18
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision import models
import torch.nn as nn
from resnet import ResNet34

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')

    parser.add_argument('--clip_quantile', default=0.99,
                        help='Clip quantile to react')

    parser.add_argument('--img_list', default=None, help='Path to image list')
    parser.add_argument("--in_dataset", choices=["cifar10", "cifar100", 'imagenet'], default="cifar100",
                        help="Which downstream task is ID.")
    parser.add_argument("--out_dataset", choices=["cifar10", "cifar100", "svhn", "isun", "lsun", "places", 'tiny_imagenet', 'imagenet-o', 'imagenet-a', 'imagenet', 'dtd', 'inat', 'open-images'], default="lsun",
                        help="Which downstream task is OOD.")
    parser.add_argument("--cls_size", type=int, default=768,
                        help="size of the class token to be used ")
    parser.add_argument("--model_name",
                        default="resnet18",
                        help="Which model to use.")
    parser.add_argument("--model_architecture_type", choices=["vit", "deit", "resnet", 'swin'],
                        default="resnet",
                        help="what type of model to use")
    parser.add_argument("--base_path", default="./",
                        help="directory where the model is saved.")
    parser.add_argument("--save_path", default="/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features",
                        help="directory where the features will be saved.")
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--neco_dim', default=100,
                        help='ETF approximative dimention')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--n_components_null_space", type=int, default=2,
                        help="Number of PCA components to be used for the null space norm")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.in_dataset == "cifar10":
        num_classes = 10
    elif args.in_dataset == "cifar100":
        num_classes = 100
    elif args.in_dataset == "imagenet":
        num_classes = 1000

    # if args.model_architecture_type == "resnet":
    #     train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_train.csv'
    #     test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_test.csv'
    #     test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'

    #     print(f" my args : {args}")
    #     args.ood_features = test_cls_tocken_path_OOD
        # ood_name = args.out_dataset
        # print(f"ood datasets: {ood_name}")
    #     model_path = f"{args.base_path}/{args.model_name}_{args.in_dataset}.pth"

    if args.model_name == 'resnet50':
        model_path = f"{args.base_path}/resnet50_{args.in_dataset}.pth"
        args.cls_size = 2048

        model = tv.models.resnet50()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        # model_layers = extract_utils.nested_children(model)
        last_layer = model_layers['fc']
        train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_train.csv'
        test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_test.csv'
        test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'

    elif args.model_name == "resnet34":
        # model = resnet_models.ResNet34(num_classes)
        # resnet_18_checkpoint = model_path
        # state_dict = torch.load(resnet_18_checkpoint)
        # model.load_state_dict(state_dict['net'], strict=False)
        # print(" acc ", state_dict['acc'])
        # # model_layers = extract_utils.nested_children(model)
        # last_layer = model_layers['linear']
        # args.cls_size = 512
        # model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model = ResNet34(num_class=100)
        # model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust final layer for 10 classes
        # model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet34_cifar10.pth'))
        model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet34_cifar100.pth'))

        train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_train.csv'
        test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.in_dataset}_test.csv'
        test_preds_path_ID = f'{args.save_path}/{args.model_name}_preds_ID_{args.in_dataset}_test.csv'
        test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'
        test_preds_path_OOD = f'{args.save_path}/{args.model_name}_preds_trained_on_{args.in_dataset}_OOD_{args.out_dataset}_test.csv'
    elif args.model_name == "resnet18":
        model = ResNet18(num_classes)
        model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet18_cifar100.pth'))
        # model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/model.pth'))
        # resnet_18_checkpoint = model_path
        # print(f" model path {model_path}")
        # state_dict = torch.load(resnet_18_checkpoint)
        # model.load_state_dict(state_dict['net'], strict=False)
        # print(" acc ", state_dict['acc'])
        args.cls_size = 512
        # model_layers = extract_utils.nested_children(model)
        # last_layer = model_layers['linear']
        # train_cls_tocken_path_ID = f'{args.save_path}/{args.in_dataset}_train.npy'
        # test_cls_tocken_path_ID = f'{args.save_path}/{args.in_dataset}.npy'
        # test_cls_tocken_path_OOD = f'{args.save_path}/{args.out_dataset}.npy'
        last_layer = model.linear
        bias = last_layer.bias
        bias.requires_grad = False
        bias = bias.detach().cpu().numpy()
        weight = last_layer.weight
        weight.requires_grad = False
        weight = weight.detach().cpu().numpy()

        # Save the weights and bias as variables
        variables = {'weight': weight, 'bias': bias}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract the fully connected layer (fc) weights and bias


    # Specify the filename for the pickle file
    # pickle_filename = 'model_weights_biases.pkl'

    # # Save the variables to the pickle file
    # with open(pickle_filename, 'wb') as f:
    #     pickle.dump(variables, f)
    # print(f'{weight.shape=}, {bias.shape=}')

 ################################################################################################################################################################################################################################################################
    print('load features')
    ID_train_path = f'{args.save_path}/{args.in_dataset}_res18_train.npy'
    train_labels_path = f'{args.save_path}/{args.in_dataset}_res18_train_label.npy'
    # test_labels_path = f'{args.save_path}/{args.in_dataset}_test_label.npy'
    test_cls_tocken_path_ID = f'{args.save_path}/{args.in_dataset}_res18_test.npy'
    test_cls_tocken_path_OOD = f'{args.save_path}/{args.out_dataset}_resnet18_ci100.npy'
    feature_id_train = np.load(ID_train_path)
    train_labels = np.load(train_labels_path)
    # test_labels = np.load(test_labels_path)
    feature_id_val = np.load(test_cls_tocken_path_ID)
    feature_ood = np.load(test_cls_tocken_path_OOD)
    weight = model.linear.weight
    weight = weight.detach().cpu().numpy()
    bias = model.linear.bias
    bias = bias.detach().cpu().numpy()
    # Print feature shapes for verification
    print(f'{feature_id_train.shape=}, {feature_id_val.shape=}, {feature_ood.shape=}')
    print(f"My args: {args}")

    # Extract weight (w) and bias (b) from the model
    print('Computing logits...')
    logit_id_train = feature_id_train @ weight.T + bias
    logit_id_val = feature_id_val @ weight.T + bias
    logit_ood = feature_ood @ weight.T + bias

    # test_transform = T.Compose([
    # T.Resize((32, 32)),
    # # T.CenterCrop(32),
    # T.ToTensor(),
    # T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    # cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    # cifar10_loader = DataLoader(cifar10_testset, batch_size=500, shuffle=True)


    # Compute softmax
    print('Computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_ood = softmax(logit_ood, axis=-1)

    # Compute u
    u = -np.matmul(pinv(weight), bias)

    # Print results for verification
    print(f'Softmax shapes: {softmax_id_train.shape=}, {softmax_id_val.shape=}, {softmax_ood.shape=}')
    # print(f'Computed u: {u}')
    ood_name = args.out_dataset
    print(f"ood datasets: {ood_name}")

    # #---------------------------------------
    method = 'MSP'   #num_fp = 20047
    print(f'\n{method}')
    ood_methods.msp(softmax_id_val, softmax_ood, ood_name)
    # ---------------------------------------
    method = 'MaxLogit'         #num_fp = 21876
    print(f'\n{method}')
    ood_methods.maxLogit(logit_id_val, logit_ood, ood_name)
    #---------------------------------------
    method = 'Energy'     #num_fp = 22202
    print(f'\n{method}')
    ood_methods.energy(logit_id_val, logit_ood, ood_name)
    # ---------------------------------------
    method = 'Energy+React'
    print(f'\n{method}')
    result = []
    thresh = 0.99
    if args.model_architecture_type == "vit":
        thresh = 0.99
    if args.model_architecture_type == "resnet":
        thresh = 0.9
    if args.model_architecture_type == "swin":
        thresh = 0.95
    clip = np.quantile(feature_id_train, thresh)
    ood_methods.react(feature_id_val, feature_ood, clip, weight, bias, ood_name)
    # ---------------------------------------
    method = 'ViM'   #10750
    print(f'\n{method}')
    ood_methods.vim(feature_id_train, feature_id_val, feature_ood, logit_id_train,
                    logit_id_val, logit_ood, ood_name, args.model_architecture_type, args.model_name, u)

    # ---------------------------------------
    method = 'NECO'  #17598
    print(f'\n{method}')
    ood_methods.neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood,
                     model_architecture_type=args.model_architecture_type, neco_dim=args.neco_dim)

    # ---------------------------------------
    method = 'Residual'
    print(f'\n{method}')
    ood_methods.residual(feature_id_train, feature_id_val,
                         feature_ood, args.model_architecture_type, u, ood_name)
    # ---------------------------------------
    method = 'GradNorm'
    print(f'\n{method}')
    result = []
    ood_methods.gradNorm(feature_id_val, feature_ood,
                         ood_name, num_classes, weight, bias)
    # ---------------------------------------
    method = 'Mahalanobis'
    print(f'\n{method}')
    ood_methods.mahalanobis(feature_id_train, train_labels,
                            feature_id_val, feature_ood, ood_name, num_classes)
    # ---------------------------------------
    method = 'KL-Matching'   # 19886
    print(f'\n{method}')
    ood_methods.kl_matching(softmax_id_train, softmax_id_val,
                            softmax_ood, ood_name, num_classes)
    
    #--------------------------------------    method = 'Multi_Proj'  #39
    # ood_methods.multiple_proj(  feature_id_val, test_labels, feature_ood, ood_name, model, device, n_components=100)


if __name__ == '__main__':
    main()

##When ID=CIFAR10, OOD=cifar100 and model=resnet34
# MSP: inat auroc 79.16%, fpr 83.51%
# Max_Logit: inat auroc 82.08%, fpr 76.88%
# Energy: inat auroc 82.08%, fpr 76.29%
# Energy+React inat auroc 82.58%, fpr 75.14%
# ViM: inat auroc 82.78%, fpr 71.98%
# NECO: auroc 81.24%, fpr 71.98%
# Residual auroc  62.50%  fpr 88.10%
# GradNorm auroc 76.15%,  fpr 81.53%
# Mahalanobis  auroc58.85%  fpr  90.95%
# KL-matching  auroc  69.97    fpr 83.25  
# Ours  auroc  96.6 fpr 17.48
#  
# When ID=CIFAR100, OOD=cifar10 and model=resnet34

# MSP: inat auroc 79.16%, fpr 83.51%
# Max_Logit: inat auroc 82.08%, fpr 76.88%
# Energy: inat auroc 82.08%, fpr 76.29%
# Energy+React inat auroc 82.58%, fpr 75.14%
# ViM: inat auroc 82.78%, fpr 71.98%
# NECO: auroc 81.24%, fpr 71.98%
# Residual auroc  62.50%  fpr 88.10%
# GradNorm auroc 76.15%,  fpr 81.53%
# Mahalanobis  auroc58.85%  fpr  90.95%
# KL-matching  auroc  69.97    fpr 83.25  
# Ours  auroc  96.6 fpr 17.48

# When ID= cifar10
#  and ood=dtd for resnet34
# For texture
# MSP: dtd auroc 81.88%, fpr 72.89%
# Max_Logit: dtd auroc 81.70%, fpr 66.81%
# Energy: dtd auroc 81.72%, fpr 66.76%
#Energy+React: dtd auroc 79.68%, fpr 66.33%
# ViM: dtd auroc 91.31%, fpr 41.40%
#  NECO: auroc 92.71%, fpr 37.96%
# Residual: dtd auroc 91.56%, fpr 41.13%
# GradNorm:  auroc 67.72%, fpr 78.95%
# Mahalanobis: auroc 92.31%, fpr 42.04%
# KL-Matching: dtd auroc 72.27%, fpr 72.46%


# For svhn
# MSP: svhn auroc 85.36%, fpr 74.30%
# Max_Logit: svhn auroc 85.02%, fpr 73.76%
# Energy: svhn auroc 85.01%, fpr 74.28%
# Energy+React: svhn auroc 82.23%, fpr 73.05%
# ViM: svhn auroc 91.24%, fpr 50.71%
#  NECO: auroc 92.18%, fpr 46.68%
# Residual: svhn auroc 89.87%, fpr 47.98%
# GradNorm:  auroc 69.24%, fpr 84.90%
# Mahalanobis: auroc 90.07%, fpr 50.01%
# KL-Matching: svhn auroc 75.95%, fpr 74.24%

#For places
# MSP: places auroc 84.55%, fpr 70.78%
# Max_Logit: places auroc 87.19%, fpr 57.09%
# Energy: places auroc 87.26%, fpr 56.75%
# ReAct: places auroc 85.07%, fpr 57.73%
# ViM: places auroc 86.04%, fpr 68.71%
#  NECO: auroc 88.44%, fpr 54.18%
# Residual: places auroc 78.44%, fpr 82.15%
# GradNorm:  auroc 76.29%, fpr 75.62%
# Mahalanobis: auroc 79.43%, fpr 83.00%
# KL-Matching: places auroc 71.22%, fpr 70.67%

#For inaturalist
# MSP: inat auroc 80.78%, fpr 76.05%
# Max_Logit: inat auroc 79.90%, fpr 73.96%
# Energy: inat auroc 79.89%, fpr 74.29%
# Energy+React: inat auroc 76.15%, fpr 73.19%
# ViM: inat auroc 85.53%, fpr 62.45%
#  NECO: auroc 89.03%, fpr 54.78%
# Residual: inat auroc 84.98%, fpr 64.02%
# GradNorm:  auroc 69.55%, fpr 86.95%
# Mahalanobis: auroc 86.25%, fpr 63.84%
# KL-Matching: inat auroc 63.57%, fpr 75.53%

# For isun
# MSP: isun auroc 90.99%, fpr 57.19%
# Max_Logit: isun auroc 93.55%, fpr 37.15%
# Energy: isun auroc 93.64%, fpr 36.38%
# Energy+React: isun auroc 92.80%, fpr 37.43%
# ViM: isun auroc 93.04%, fpr 42.53%
#  NECO: auroc 94.75%, fpr 28.09%
# Residual: isun auroc 86.75%, fpr 67.00%
# GradNorm:  auroc 86.11%, fpr 59.81%
# Mahalanobis: auroc 88.32%, fpr 63.04%
# KL-Matching: isun auroc 80.25%, fpr 56.48%
# Our : isun auroc 98.97, fpr 3.05


#For lsun
# MSP: lsun auroc 94.59%, fpr 44.07%
# Max_Logit: lsun auroc 97.78%, fpr 13.43%
# Energy: lsun auroc 97.92%, fpr 12.77%
# Energy+React: lsun auroc 97.43%, fpr 14.79%
# ViM: lsun auroc 86.19%, fpr 96.70%
#  NECO: auroc 95.52%, fpr 27.27%
# Residual: lsun auroc 45.98%, fpr 99.69%
# GradNorm:  auroc 98.42%, fpr 8.21%
# KL-Matching: lsun auroc 89.08%, fpr 45.81%


# When id = cifar100, model=resnet34
# For lsun
# MSP: lsun auroc 80.33%, fpr 73.37%
# Max_Logit: lsun auroc 88.86%, fpr 57.68%
# Energy: lsun auroc 89.37%, fpr 55.82%
# Energy+React: lsun auroc 85.87%, fpr 66.20%
# ViM: lsun auroc 67.67%, fpr 98.64%
#  NECO: auroc 35.78%, fpr 98.82%
# Residual: lsun auroc 9.29%, fpr 99.98%
# GradNorm:  auroc 90.67%, fpr 44.24%
# Mahalanobis: auroc 14.70%, fpr 99.99%
# KL-Matching: lsun auroc 79.31%, fpr 74.42%
# Ours: auroc 82.29, fpr 58.18 

# For isun
# MSP: isun auroc 81.06%, fpr 73.94%
# Max_Logit: isun auroc 88.81%, fpr 57.69%
# Energy: isun auroc 89.27%, fpr 55.36%
# Energy+React: isun auroc 89.61%, fpr 53.84%
# ViM: isun auroc 84.95%, fpr 77.05%
# NECO: auroc 83.29%, fpr 64.87%
# Residual: isun auroc 39.54%, fpr 98.03%
# GradNorm:  auroc 78.76%, fpr 67.44%
# Mahalanobis: auroc 47.69%, fpr 97.65%
# KL-Matching: isun auroc 81.11%, fpr 72.48%
# Ours: auroc 94.36, fpr 24.54

# For dtd
# MSP: dtd auroc 69.42%, fpr 88.99%
# Max_Logit: dtd auroc 70.17%, fpr 90.67%
# Energy: dtd auroc 70.00%, fpr 91.15%
# Energy+React: dtd auroc 75.65%, fpr 87.32%
# ViM: dtd auroc 86.90%, fpr 50.23%
#  NECO: auroc 87.30%, fpr 46.86%
# Residual: dtd auroc 80.09%, fpr 54.73%
# GradNorm:  auroc 56.37%, fpr 87.23%
# Mahalanobis: auroc 83.24%, fpr 50.30%
# KL-Matching: dtd auroc 69.25%, fpr 86.31%
# Ours: auroc 28.01 ,fpr 95.81

# For inat:
# MSP: inat auroc 72.61%, fpr 87.95%
# Max_Logit: inat auroc 72.40%, fpr 88.50%
# Energy: inat auroc 72.20%, fpr 88.93%
# Energy+React: inat auroc 76.37%, fpr 83.97%
# ViM: inat auroc 77.46%, fpr 78.53%
#  NECO: auroc 63.42%, fpr 92.23%
# Residual: inat auroc 61.27%, fpr 88.21%
# GradNorm:  auroc 37.04%, fpr 97.76%
# Mahalanobis: auroc 68.47%, fpr 83.88%
# KL-Matching: inat auroc 72.24%, fpr 87.81%
# Ours: auroc 95.44,  fpr  24.54 
 
# For places:
# MSP: places auroc 68.11%, fpr 87.77%
# Max_Logit: places auroc 71.23%, fpr 84.65%
# Energy: places auroc 71.32%, fpr 84.15%
# Energy+React: places auroc 72.79%, fpr 81.90%
# ViM: places auroc 72.05%, fpr 83.89%
#  NECO: auroc 59.86%, fpr 91.24%
# Residual: places auroc 51.98%, fpr 94.58%
# GradNorm:  auroc 51.95%, fpr 94.36%
# Mahalanobis: auroc 56.62%, fpr 93.75%
# KL-Matching: places auroc 66.65%, fpr 86.47%
# Ours: places auroc 88.95, fpr 57.98

# For svhn

# MSP: svhn auroc 72.26%, fpr 86.36%
# Max_Logit: svhn auroc 72.97%, fpr 93.38%
# Energy: svhn auroc 72.62%, fpr 95.17%
# Energy+React: svhn auroc 75.90%, fpr 94.71%
# ViM: svhn auroc 85.48%, fpr 67.21%
# NECO: auroc 84.19%, fpr 70.44%
# Residual: svhn auroc 77.33%, fpr 77.90%
# GradNorm:  auroc 61.71%, fpr 93.32%
# Mahalanobis: auroc 84.35%, fpr 67.28%
# KL-Matching: svhn auroc 74.59%, fpr 80.34%
# Ours: 99.05 auroc, fpr 5.25


# When id = cifar100, model=resnet18

# For svhn
# MSP: svhn auroc 73.12%, fpr 83.66%
# Max_Logit: svhn auroc 77.49%, fpr 84.93%
# Energy: svhn auroc 77.59%, fpr 85.92%
# Energy+React:svhn auroc 84.69%, fpr 79.04%
# ViM: svhn auroc 91.90%, fpr 44.61%
# NECO: auroc 91.33%, fpr 43.55%
# Residual: svhn auroc 76.31%, fpr 73.08%
# GradNorm:  auroc 79.42%, fpr 74.37%
# Mahalanobis: auroc 80.71%, fpr 70.31%
# KL-Matching: svhn auroc 74.65%, fpr 81.99%
# Ours: auroc:98.64, fpr:7.14

# for iNaturlist
# MSP: inat auroc 82.73%, fpr 69.81%
# Max_Logit: inat auroc 87.08%, fpr 63.88%
# Energy: inat auroc 87.31%, fpr 62.66%
# Energy+React: inat auroc 87.78%, fpr 60.60%
# ViM: inat auroc 90.34%, fpr 49.32%
# NECO: auroc 85.48%, fpr 63.59%
# Residual: inat auroc 62.91%, fpr 86.34%
# GradNorm:  auroc 78.77%, fpr 64.18%
# Mahalanobis: auroc 65.66%, fpr 86.47%
# KL-Matching: inat auroc 82.75%, fpr 70.73%
# Ours: auroc:97.37, fpr:14.13

# for places
# MSP: places auroc 81.58%, fpr 69.76%
# Max_Logit: places auroc 85.93%, fpr 64.14%
# Energy: places auroc 86.18%, fpr 62.48%
# Energy+React: places auroc 86.59%, fpr 61.55%
# ViM: places auroc 88.94%, fpr 54.33%
# NECO: auroc 84.37%, fpr 65.36%
# Residual: places auroc 60.86%, fpr 87.48%
# GradNorm:  auroc 77.95%, fpr 64.79%
# Mahalanobis: auroc 63.65%, fpr 87.95%
# KL-Matching: places auroc 81.35%, fpr 71.28%
# Ours: auroc:97.03, fpr:15.63

# for Textures
# MSP: dtd auroc 78.95%, fpr 76.54%
# Max_Logit: dtd auroc 81.78%, fpr 74.54%
# Energy: dtd auroc 81.71%, fpr 73.99%
# Energy+React: dtd auroc 84.79%, fpr 67.20%
# ViM: dtd auroc 92.04%, fpr 37.66%
# NECO: auroc 89.82%, fpr 44.88%
# Residual: dtd auroc 74.95%, fpr 64.66%
# GradNorm:  auroc 71.19%, fpr 71.76%
# Mahalanobis: auroc 77.70%, fpr 62.77%
# KL-Matching: dtd auroc 79.40%, fpr 75.53%
# Ours: auroc:97.80, fpr:14.42

# for iSUN
# MSP: isun auroc 80.53%, fpr 71.62%
# Max_Logit: isun auroc 85.29%, fpr 66.20%
# Energy: isun auroc 85.59%, fpr 64.95%
# Energy+React: isun auroc 86.06%, fpr 64.38%
# ViM: isun auroc 87.33%, fpr 61.56%
# NECO: auroc 81.52%, fpr 71.57%
# Residual: isun auroc 57.63%, fpr 91.08%
# GradNorm:  auroc 77.85%, fpr 67.36%
# Mahalanobis: auroc 60.40%, fpr 91.92%
# KL-Matching: isun auroc 80.15%, fpr 73.56%
# Ours: auroc:96.88, fpr:16.44

# for LSUN
# MSP: lsun auroc 82.92%, fpr 68.05%
# Max_Logit: lsun auroc 87.39%, fpr 60.54%
# Energy: lsun auroc 87.70%, fpr 58.62%
# Energy+React:lsun auroc 88.39%, fpr 55.91%
# ViM: lsun auroc 90.67%, fpr 45.85%
# NECO: auroc 87.52%, fpr 54.46%
# Residual: lsun auroc 63.13%, fpr 82.64%
# GradNorm:  auroc 80.07%, fpr 59.63%
# Mahalanobis: auroc 66.60%, fpr 82.88%
# KL-Matching: lsun auroc 82.87%, fpr 69.37%
# Ours: auroc:97.75, fpr:12.61
