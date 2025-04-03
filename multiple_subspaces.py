import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision import models
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torchvision
from utils_ood import fpr_recall, auc 
from resnet import ResNet34
import time

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)  #  25088 for caltech and 512 for rest of the datasets
        self.linear1 = nn.Linear(512*block.expansion, num_classes)    #25088 for caltech and 512 for rest of the datasets

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out5 = out5.view(out5.size(0), -1)
        out = self.linear(out5)
        out_cons = self.linear1(out5)

        return out, out_cons, out5, [out1, out2, out3, out4]


def ResNet18(num_class):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_class)

# Transformations for training and testing
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = T.Compose([
    T.Resize((32, 32)),
    # T.CenterCrop(32),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

# Load CIFAR-10 (ID) and SVHN (OOD) datasets
cifar10_trainset = datasets.CIFAR10(root='/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/CIFAR10/id_data', train=True, download=True, transform=train_transform)
cifar10_testset = datasets.CIFAR10(root='/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/CIFAR10/id_data', train=False, download=True, transform=test_transform)

svhn_dataset = datasets.SVHN(root='/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/CIFAR10/ood_data', split='test', download=True, transform=test_transform)

cifar10_testloader = DataLoader(cifar10_testset, batch_size=1000, shuffle=True)
cifar10_trainloader = DataLoader(cifar10_trainset, batch_size=1000, shuffle=True)
svhn_loader = DataLoader(svhn_dataset, batch_size=1000, shuffle=True)
svhn_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/svhn_resnet18.npy')

cifar100_trainset = torchvision.datasets.CIFAR100(root='/usr/local/home/sgchr/Documents/OOD/Multiple_spaces', train=True, transform=train_transform, download=True)
cifar100_trainloader = DataLoader(cifar100_trainset, batch_size=1000, shuffle=True)
cifar100_testset = datasets.CIFAR100(root='/usr/local/home/sgchr/Documents/OOD/Multiple_spaces', train=False, download=True, transform=test_transform)
cifar100_testloader = DataLoader(cifar100_testset, batch_size=1000, shuffle=True)
dtd_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/dtd/images", transform=test_transform),
                 batch_size=500, shuffle=False)  # 47 folders and 5640 images
# # dtd_features = np.load('/cluster/pixstor/madrias-lab/Shreen/My_Try/PK_files/dtd_penul_resnet18_feat.npy')

inat_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/iNaturalist",
                 transform=test_transform), batch_size=500, shuffle=False)
# # inat_features = np.load('/cluster/pixstor/madrias-lab/Shreen/My_Try/PK_files/inat_penul_resnet18_feat.npy')

places_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/Places",
                 transform=test_transform), batch_size=500, shuffle=False)
# places_features = np.load('/cluster/pixstor/madrias-lab/Shreen/My_Try/PK_files/places_penul_resnet18_feat.npy')

lsun_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/CIFAR10/ood_data/LSUN",
                 transform=test_transform), batch_size=500, shuffle=False)
isun_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/CIFAR10/ood_data/iSUN",
                 transform=test_transform), batch_size=500, shuffle=False)
#
# Assuming ResNet18 is already defined as provided

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(num_class=10).to(device)
# model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet18_cifar100.pth'))

model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/model.pth'))
# model = ResNet34(num_class=100).to(device)
# model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet34_cifar100.pth'))
# model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
# model.fc = nn.Linear(model.fc.in_features, 100)  # Adjust final layer for 10 classes
# model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet34_cifar100.pth'))
# model = model.to(device)
# model.eval()

def calculate_accuracy(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)[0]  #[0]  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            total += labels.size(0)  # Count the total number of samples
            correct += (predicted == labels).sum().item()  # Count correctly predicted samples

    accuracy = correct / total * 100  # Calculate accuracy as a percentage
    return accuracy

# accuracy = calculate_accuracy(model, cifar100_testloader, device)
# print(f"Accuracy on test dataset: {accuracy:.2f}%") 


# Function to extract penultimate features
def extract_features(loader):
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, _, penultimate_features, _ = model(inputs)
            # penultimate_features = feature_extractor(inputs)
            features.append(penultimate_features.cpu())
            labels.append(targets.cpu())
    return torch.cat(features), torch.cat(labels)

# Extract features for CIFAR-10 (ID) and SVHN (OOD)
# id_features, id_labels = extract_features(cifar100_testloader)
# np.save('cifar100_res18_test.npy',id_features.numpy())
# np.save('cifar100_res18_test_label.npy',id_labels.numpy())
# ood_features, _ = extract_features(lsun_loader)
ood_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/svhn_resnet18.npy')
# np.save('lsun_resnet18_ci100.npy', ood_features)
# id_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar10_res34_test.npy')
# id_labels = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar10_res34_test_label.npy')
# id_features= np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar100_res34_test.npy')
# id_labels = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar100_res34_test_label.npy')
id_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar10_test.npy')
id_labels = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar10_test_label.npy')
# Step 1: Subspace Creation using SVD
class_subspaces = {}
for i in range(10):
    class_features = id_features[id_labels == i]
    svd = TruncatedSVD(n_components=100)   #100 for resnet18 and cifar10
    svd.fit(class_features)   #.numpy()
    class_subspaces[i] = svd.components_  # Basis vectors for class i


# Step 2: Projection and Alignment Analysis

def project_and_analyze(sample, subspaces):
    alignments = {}
    for cls, basis in subspaces.items():
        projection = sample @ basis.T @ basis  # Project onto subspace
        error = np.linalg.norm(sample - projection)  # Reconstruction error
        alignments[cls] = error
    sorted_alignments = sorted(alignments.items(), key=lambda x: x[1])
    return sorted_alignments


# Analyze alignment for ID and OOD samples
def calculate_best_errors(features, class_subspaces,):
    scores = []  # List to store best errors for the dataset
    errors = []
    for sample in features:
        alignment_errors = project_and_analyze(sample, class_subspaces)
        best_error = alignment_errors[0][1]  # Smallest error
        # best_error = sum(alignment_errors[i+1][1] - alignment_errors[i][1] for i in range(3))
        scores.append(best_error)
        errors.append(alignment_errors)
    print(f"Calculated best errors for data.")
    return scores, errors

def compute_metrics(id_features, ood_features, class_subspaces, recall=0.95):
    score_id = calculate_best_errors(id_features, class_subspaces)
    score_ood = calculate_best_errors(ood_features, class_subspaces)
    auc_ood = auc(score_ood, score_id)[0]
    fpr_ood, _ = fpr_recall(score_ood, score_id, recall)
    return auc_ood, fpr_ood 


# svd_components = [50, 100, 150, 200, 250, 300]  # for cifar10
# # svd_components = [10, 20, 30, 40, 50, 60, 70, 80]   #for cifar100
# # 
# # Lists to store results
# auroc_values = []
# fpr95_values = []
# variance_retained = []
# execution_times = []

# # Compute metrics for different subspace sizes
# for n_components in svd_components:
#     start_time = time.time()

#     # Step 1: Subspace Creation using SVD
#     class_subspaces = {}
#     variance_explained = []
    
#     for i in range(10):  # CIFAR-10 has 10 classes
#         class_features = id_features[id_labels == i]
#         class_features_centered = class_features - class_features.mean(axis=0)
#         svd = TruncatedSVD(n_components=n_components)
#         svd.fit(class_features_centered)
#         class_subspaces[i] = svd.components_
#         variance_explained.append(sum(svd.explained_variance_ratio_))  # Track variance retained
    
#     # Compute AUROC and FPR95 for this subspace configuration
#     auc_ood, fpr_ood = compute_metrics(id_features, ood_features, class_subspaces, recall=0.95)

#     # Record execution time
#     elapsed_time = time.time() - start_time

#     # Store results
#     auroc_values.append(auc_ood)
#     fpr95_values.append(fpr_ood)
#     variance_retained.append(np.mean(variance_explained))  # Average variance retained across classes
#     execution_times.append(elapsed_time)

#     print(f"Components: {n_components}, AUROC: {auc_ood:.2%}, FPR95: {fpr_ood:.2%}, Variance Retained: {np.mean(variance_explained):.2%}, Time: {elapsed_time:.2f}s")

# # Save results for future reference
# results_dict = {
#     "svd_components": svd_components,
#     "auroc_values": auroc_values,
#     "fpr95_values": fpr95_values,
#     "variance_retained": variance_retained,
#     "execution_times": execution_times
# }

# np.save("svd_analysis_results_34.npy", results_dict)



# recall=0.95
# method = 'multi-proj'
# # Perform analysis
# score_id = calculate_best_errors(id_features, class_subspaces) #.numpy()
# # score_id = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/cifar10_res18_100_comp_err.npy')
# score_ood = calculate_best_errors(ood_features, class_subspaces)
# auc_ood = auc(score_ood, score_id)[0]
# fpr_ood, _ = fpr_recall(score_ood, score_id, recall)
# result = dict(method='multi_proj', oodset='places', auroc=auc_ood, fpr=fpr_ood)
# np.save('cifar10_res34_300_comp_err.npy',score_id)
# np.save('cifar100_res34_300_comp_err.npy',score_ood)

# print(f'{method}:  auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
# print('hey')



# VARIANCE explained: using the formula np.sum(singular_values[:30]**2)/np.sum(singular_values**2), I am able to achieve 
# 99.3% variance explained by the 30 components and 98% with 20 components and 97% with 10. So instead of 128 try with
# 20 and 30 components once.
#In cifar10_vs_SVHN, when #_comp = 10, auroc:97.41, fpr:12.13, #_comp = 30, auroc:97.89,fpr=10.31,
# when #_comp =60 auroc:99.12 ,fpr: 4.63,  when #_comp=80 auroc:99.44, fpr:2.56, when #_comp=100 auroc: 99.68, fpr: 0.87

#RESULTS#
#When looking at the difference of the first two errors amd setting threshold:
# For threshold = 0.3:                      For threshold=0.25        
# Results for ID (CIFAR-10) data:           Results for ID (CIFAR-10) data: 
# {'strict': 7741, 'none': 2259}             {'strict': 8359, 'none': 1641}                        
# Results for OOD (SVHN) data:              Results for OOD (SVHN) data:                
# {'strict': 2927, 'none': 23105}            {'strict': 4657, 'none': 21375}

# when looking at the magnitude of the first error
# Results for ID (CIFAR-10) data:          
# {'ID': 9388, 'OOD': 612}                                  
# Results for OOD (SVHN) data:                          
# {'ID': 139, 'OOD': 25893}    out of 26032 samples
# Result for Places data:
# {'ID': 1087, 'OOD': 8913}  out of 10k samples
# Result for SUN data:
# {'ID': 927, 'OOD' : 9073}   out of 10k samples
# Result for inat:
# {'ID : 1477, 'OOD' : 8523}  out of 10k samples
# Resukt for dtd:
# {'ID : 183, 'OOD': 5457}  out of 5640 samples

# Following are the FPR values:
# For SVHN
# {'FPR': 0.0014, 'AUROC': 0.9987001402120467, 'DTERR': 0.014986678507992896, 'AUIN': 0.9995005280150328, 'AUOUT': 0.9964197105288426}
# For Places 
# {'FPR': 0.2185, 'AUROC': 0.967346055, 'DTERR': 0.0833, 'AUIN': 0.9744702426771864, 'AUOUT': 0.9568297181884142} 
# For Textures
# {'FPR': 0.0218, 'AUROC': 0.9923002304964539, 'DTERR': 0.029603580562659847, 'AUIN': 0.9899391647349521, 'AUOUT': 0.994834064374813}
# For sun50
# {'FPR': 0.1889, 'AUROC': 0.9711405200000001, 'DTERR': 0.0742, 'AUIN': 0.9777321418883865, 'AUOUT': 0.9610769266008924}
# For inat
# {'FPR': 0.2388, 'AUROC': 0.96088969, 'DTERR': 0.1004, 'AUIN': 0.9675224697853992, 'AUOUT': 0.9539673501649392}
# Papers to compare with
# 1)  knn    2) energy   3) ODIN  4) React  5) revisit PCA-based  6) kernel PCA  7) NECO  8) Mahalanobis    9) ReAct

# With components=100 in cifar10 in resnet18, 
# For places auroc:94.88%, fpr:31.86%
# For textures  auroc: 98.60  fpr:7.37
# For iSUN  auroc: 98.54   fpr: 6.69
# For LSUN auroc: 99.26  fpr:3.14
# For iNatural auroc: 93.71  fpr:33.46
# for cifar100 auroc:95.56   fpr:26.55

#With  in cifar10 in resnet34,
# For places auroc:84.88%, fpr:51.86%   no_components=20
# For places auroc:92.95%, fpr:40.36.%   no_components=100
# For places auroc:95.71%, fpr:30.61%   no_components=150
# For places auroc:97.37, fpr:19.11%    no_components=200
# For places auroc:98.34, fpr:8.67%    no_components=250
# IF going with 150 compo
# For iNatural auroc: 96.53  fpr:22.95
# For svhn auroc: 99.09%, fpr:4.28%
# For dtd auroc:98.71,  fpr:6.13
# For lsun: auroc:98.89, fpr:6.10
# For isun:   auroc:98.98  ,fpr:2.91
#For places auroc: 95.71, fpr: 30.61


# With components=300 in ID =cifar10 and OOD =cifar100 and resnet34
# auroc 97.52, fpr 17.48


####Code to produce the image of bar charts to include in the main diagram
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Plot distributions
# plt.figure(figsize=(8, 5))
# sns.histplot(score_id, bins=50, color='blue', alpha=0.6, label='ID (CIFAR-10)', kde=True)
# sns.histplot(score_ood, bins=50, color='red', alpha=0.6, label='OOD (Textures)', kde=True)

# # Labels and title with increased font size
# plt.xlabel("Smallest Residual Error", fontsize=19)  # Increased size
# plt.ylabel("Frequency", fontsize=19)  # Increased size
# #plt.title("Distribution of Smallest Residual Errors for ID and OOD Samples", fontsize=18, fontweight='bold')

# # Legend with increased font size
# plt.legend(fontsize=19)
# plt.xticks(fontsize=15)  # Increase x-axis tick label size
# plt.yticks(fontsize=15)  # Increase y-axis tick label size

# # Show and save the plot
# plt.show()
# plt.savefig('CIFAR_Textures_bar')

# To generate all the plots for the paper use, following format

# plt.rcParams['font.family'] = 'serif'
# plt.figure(figsize=(8, 6))

# # Plot ID error distribution (one class should be low)
# plt.plot(np.mean(id_sorted_errors, axis=0), marker='o', label='ID (CIFAR-10)', linestyle='-')

# # Plot OOD error distribution (uniform across subspaces)
# plt.plot(np.mean(ood_sorted_errors, axis=0), marker='s', label='OOD (LSUN)', linestyle='--')

# plt.xlabel('Class Subspaces', fontsize=20, fontname='serif')  # Set font
# plt.ylabel('Residual Error', fontsize=20, fontname='serif')  # Set font
# # plt.title('Residual Error Analysis Across Subspaces for ID and OOD Data', fontname='Arial')  # Title font (if needed)
# plt.legend(fontsize=17)
# plt.xticks(fontsize=17)  # Increase x-axis tick label size & set font
# plt.yticks(fontsize=17)  # Increase y-axis tick label size & set font
# plt.grid(True)
# plt.savefig('Errors_lsun_sorted_full', dpi=300)




# lsun_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/CIFAR10/ood_data/LSUN",
#                  transform=test_transform), batch_size=500, shuffle=False)






# dtd_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/dtd/images", transform=test_transform),
#                  batch_size=500, shuffle=False)
# dtd_features, _ = extract_features(dtd_loader)
# ood_sorted_errors = calculate_best_errors(dtd_features)
# plt.rcParams['font.family'] = 'serif'
# plt.figure(figsize=(8, 6))

# # Plot ID error distribution (one class should be low)
# sns.kdeplot(score_id, label='ID (CIFAR-10)', bw_adjust=1, fill=True, alpha=0.5)
# sns.kdeplot(ood_sorted_errors, label='OOD (Textures)', bw_adjust=1, fill=True, alpha=0.5)
# plt.xlabel('Detection Score', fontsize=20, fontname='serif')  # Set font
# plt.ylabel('Density', fontsize=20, fontname='serif')  # Set font
# # plt.title('Residual Error Analysis Across Subspaces for ID and OOD Data', fontname='Arial')  # Title font (if needed)
# plt.legend(fontsize=17)
# plt.xticks(fontsize=17)  # Increase x-axis tick label size & set font
# plt.yticks(fontsize=17)  # Increase y-axis tick label size & set font
# # plt.grid(True)
# plt.savefig('dtd_scores', dpi=300)


# places_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/Places",
#                  transform=test_transform), batch_size=500, shuffle=False)
# places_features, _ = extract_features(places_loader)
# ood_sorted_errors = calculate_sorted_residual_errors(places_features, class_subspaces)
# plt.rcParams['font.family'] = 'serif'
# plt.figure(figsize=(8, 6))

# # Plot ID error distribution (one class should be low)
# plt.plot(np.mean(id_sorted_errors, axis=0), marker='o', label='ID (CIFAR-10)', linestyle='-')

# # Plot OOD error distribution (uniform across subspaces)
# plt.plot(np.mean(ood_sorted_errors, axis=0), marker='s', label='OOD (Places)', linestyle='--')

# plt.xlabel('Class Subspaces', fontsize=20, fontname='serif')  # Set font
# plt.ylabel('Residual Error', fontsize=20, fontname='serif')  # Set font
# # plt.title('Residual Error Analysis Across Subspaces for ID and OOD Data', fontname='Arial')  # Title font (if needed)
# plt.legend(fontsize=17)
# plt.xticks(fontsize=17)  # Increase x-axis tick label size & set font
# plt.yticks(fontsize=17)  # Increase y-axis tick label size & set font
# plt.grid(True)
# plt.savefig('Errors_places_sorted_full', dpi=300)








