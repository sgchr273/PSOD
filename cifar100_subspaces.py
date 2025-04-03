
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision import models
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import gc
import torchvision
from tqdm import tqdm
import utils_ood as utils
from scipy.special import logsumexp
from resnet import ResNet34, ResNet18
import time
import torchvision.transforms as T
# from multiple_subspaces import cifar10_testloader


model = ResNet34(num_class=100)
model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet34_cifar100.pth'))
# model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet34_cifar10.pth'))

# model = ResNet18(num_class=10)
# model.state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet18_cifar100.pth'))
# model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/model.pth'))

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def calculate_accuracy(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)[0]  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability  outputs[0] for mobilenetv2 
            total += labels.size(0)  # Count the total number of samples
            correct += (predicted == labels).sum().item()  # Count correctly predicted samples

    accuracy = correct / total * 100  # Calculate accuracy as a percentage
    return accuracy

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

cifar100_trainset = torchvision.datasets.CIFAR100(root='/usr/local/home/sgchr/Documents/OOD/Multiple_spaces', train=True, transform=train_transform, download=True)
cifar100_trainloader = DataLoader(cifar100_trainset, batch_size=1000, shuffle=True)
cifar100_testset = datasets.CIFAR100(root='/usr/local/home/sgchr/Documents/OOD/Multiple_spaces', train=False, download=True, transform=test_transform)
cifar100_testloader = DataLoader(cifar100_testset, batch_size=1000, shuffle=True)

accuracy = calculate_accuracy(model, cifar100_testloader, device)
print(f"Accuracy on test dataset: {accuracy:.2f}%") 

# penultimate_layer = nn.Sequential(*list(model.children())[:-1])
# model.eval()

def extract_features_in_chunks(loader, model, device, save_path="features_labels", chunk_size=5000):
    os.makedirs(save_path, exist_ok=True)  # Create directory to store chunks
    features, labels = [], []
    count = 0
    chunk_index = 0

    model.to(device).eval()  # Ensure the model is in eval mode and on the correct device

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            penultimate_features = model(inputs)[1].squeeze(-1).squeeze(-1)  # model(inputs)[1] for mobilenetv2

            features.append(penultimate_features)
            labels.append(targets)
            count += inputs.size(0)

            if count >= chunk_size:  # Save when chunk is full
                features_gpu = torch.cat(features)  # Keep computation on GPU
                labels_gpu = torch.cat(labels)

                # Move to CPU in one step to minimize transfers
                features_np = features_gpu.cpu().numpy()
                labels_np = labels_gpu.cpu().numpy()

                np.save(os.path.join(save_path, f"features_chunk_{chunk_index}.npy"), features_np)
                np.save(os.path.join(save_path, f"labels_chunk_{chunk_index}.npy"), labels_np)

                print(f"Saved chunk {chunk_index}: {count} samples")

                # Reset buffers for next chunk
                features, labels = [], []
                count = 0
                chunk_index += 1

    # Save any remaining data that didn't fill a full chunk
    if features:
        features_gpu = torch.cat(features)
        labels_gpu = torch.cat(labels)
        features_np = features_gpu.cpu().numpy()
        labels_np = labels_gpu.cpu().numpy()

        np.save(os.path.join(save_path, f"features_chunk_{chunk_index}.npy"), features_np)
        np.save(os.path.join(save_path, f"labels_chunk_{chunk_index}.npy"), labels_np)
        print(f"Saved final chunk {chunk_index}: {len(features_np)} samples")

    print(f"All feature chunks saved in '{save_path}/'")


# extract_features_in_chunks(val_loader, model, device, save_path="/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features", chunk_size=5000)

def load_and_concatenate_npy_files(folder_path):
    # Get all feature chunk files
    feature_files = sorted([f for f in os.listdir(folder_path) if "features_chunk" in f])
    
    # Load and concatenate all feature chunks
    feature_arrays = [np.load(os.path.join(folder_path, f)) for f in feature_files]
    full_features = np.concatenate(feature_arrays, axis=0)
    
    print(f"Concatenated array shape: {full_features.shape}")
    return full_features

# Specify the folder where chunks are stored
folder_path = "/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features"  # Change this to your actual path

# Load and concatenate all feature files
# full_feature_array = load_and_concatenate_npy_files(folder_path)

# dtd_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/dtd/images", transform=transform_test_largescale),
#                  batch_size=500, shuffle=False)
# dtd_features= np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/dtd_resnet50.npy')

inat_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/iNaturalist",
                 transform=test_transform), batch_size=500, shuffle=False)
# inat_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/inat_resnet50.npy')
# places_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/Places",
                #  transform=transform_test_largescale), batch_size=500, shuffle=False)
# places_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/places_resnet50.npy')
# sun_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Imagenet/ood_data/SUN",
                #  transform=transform_test_largescale), batch_size=500, shuffle=False)
# sun_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/sun_resnet50.npy')

def multiple_proj( id_features, id_labels, ood_loader, ood_features,  name, device,  n_components=40):
    method = 'Multi_proj'
    """
    Perform the entire pipeline of feature extraction, subspace creation, 
    projection, and reconstruction error calculation in one function.
    """
    # Step 1: Extract Features
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, _, penultimate_features, _ = model(inputs)
                # penultimate_features= penultimate_layer(inputs).squeeze(-1).squeeze(-1)
                features.append(penultimate_features.cpu())
                labels.append(targets.cpu())
        return torch.cat(features), torch.cat(labels)
    
    print("Extracting ID and OOD features...")
    # id_features, id_labels = extract_features(id_loader)
    # ood_features, _ = extract_features(ood_loader)
    # np.save('dtd_mobilenet.npy', np.array(ood_features))
    
    # Step 2: Create Subspaces
    def create_subspaces(features, labels, n_components):
        subspaces = {}
        unique_labels = torch.unique(labels)  # Get unique classes

        for i in unique_labels:
            class_features = features[labels == i]  # Extract class-specific features

            # Compute SVD on GPU
            U, S, Vh = torch.linalg.svd(class_features, full_matrices=False)

            # Take the top `n_components` right singular vectors (Vh.T)
            subspaces[int(i.item())] = Vh[:n_components].to(device).float()  # Basis vectors

        return subspaces

    # Move ID features and labels to GPU
    # id_features = torch.tensor(id_features, dtype=torch.float32).to(device)
    # id_labels = torch.tensor(id_labels, dtype=torch.int64).to(device)
    # ood_features = torch.tensor(ood_features, dtype=torch.float32).to(device)

    print("Creating subspaces for ID classes...")
    # subspaces = create_subspaces(id_features, id_labels, n_components=20)

    # Function to project and analyze samples using GPU
    def project_and_analyze(sample, subspaces):
        sample = sample.to(device).float()  # Ensure sample is on GPU
        errors = {}

        for cls, basis in subspaces.items():
            projection = (sample @ basis.T) @ basis  # Projection onto subspace
            error = torch.norm(sample - projection)  # Compute residual error
            errors[cls] = error.item()

        sorted_errors = sorted(errors.items(), key=lambda x: x[1])  # Sort by error
        return sorted_errors

    # Function to compute best reconstruction errors in a vectorized way
    def calculate_best_errors(features, subspaces, batch_size=1024):
            subspaces = {key: torch.tensor(value, dtype=torch.float32).to(device) for key, value in subspaces.items()}
            features = torch.tensor(features, dtype=torch.float32).to(device)
            features = features.float().to(device)  # Ensure all features are float32
            scores = []
            
            with torch.no_grad():  # Reduce memory usage
                num_samples = features.shape[0]
                
                for start in range(0, num_samples, batch_size):
                    end = min(start + batch_size, num_samples)
                    batch = features[start:end]

                    # Batch computation of errors
                    batch_errors = torch.full((batch.shape[0], len(subspaces)), float("inf"), device=device)

                    for cls, basis in subspaces.items():
                        projection = (batch @ basis.T) @ basis  # Batch projection
                        error = torch.norm(batch - projection, dim=1)  # Compute residual error

                        batch_errors[:, cls] = error  # Store errors for each class

                    best_errors = batch_errors.min(dim=1)[0]  # Get minimum error per sample
                    scores.append(best_errors)

            scores = torch.cat(scores).cpu().numpy()  # Move back to CPU for processing
            print(f"Calculated best errors for data.")
            return scores

    
    def compute_metrics(id_features, ood_features, class_subspaces, recall=0.95):
        score_id = calculate_best_errors(id_features, class_subspaces, batch_size=4096)
        score_ood = calculate_best_errors(ood_features, class_subspaces, batch_size=4096)
        auc_ood = utils.auc(score_ood, score_id)[0]
        fpr_ood, _ = utils.fpr_recall(score_ood, score_id, recall)
        return auc_ood, fpr_ood

    # svd_components = [50, 100, 150, 200, 250, 300] 
    # svd_components = [5,10,15,20,25,30,35,40,45,50]  
    svd_components = [5, 10, 15, 20 ,25, 30, 35, 40]  #for cifar100
    # svd_components = [20,40,60,80,100,120,140,150]  #for cifar10
    # Lists to store results
    auroc_values = []
    fpr95_values = []
    variance_retained = []
    execution_times = []

    # Compute metrics for different subspace sizes
    for n_components in svd_components:
        start_time = time.time()

        # Step 1: Subspace Creation using SVD
        class_subspaces = {}
        variance_explained = []
        
        for i in range(100):  # CIFAR-10 has 10 classes
            class_features = id_features[id_labels == i]
            # class_features = class_features - class_features.mean(axis=0)
            svd = TruncatedSVD(n_components=n_components)
            svd.fit(class_features)
            class_subspaces[i] = svd.components_
            variance_explained.append(sum(svd.explained_variance_ratio_))  # Track variance retained
        
        # Compute AUROC and FPR95 for this subspace configuration
        auc_ood, fpr_ood = compute_metrics(id_features, ood_features, class_subspaces)

        # Record execution time
        elapsed_time = time.time() - start_time

        # Store results
        auroc_values.append(auc_ood)
        fpr95_values.append(fpr_ood)
        variance_retained.append(np.mean(variance_explained))  # Average variance retained across classes
        execution_times.append(elapsed_time)

        print(f"Components: {n_components}, AUROC: {auc_ood:.2%}, FPR95: {fpr_ood:.2%}, Variance Retained: {np.mean(variance_explained):.2%}, Time: {elapsed_time:.2f}s")

    # Save results for future reference
    results_dict = {
        "svd_components": svd_components,
        "auroc_values": auroc_values,
        "fpr95_values": fpr95_values,
        "variance_retained": variance_retained,
        "execution_times": execution_times
    }

    # np.save("svd_analysis_results_res34_places.npy", results_dict)

    print("Analyzing features...")
    # score_ood = calculate_best_errors(ood_features, "OOD Data", batch_size=4096)
    # np.save('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Best_errors/dtd_17_comp_samllest_err.npy', score_ood)
    print('Done with OOD')


id_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar100_res34_test.npy')
id_labels = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar100_res34_test_label.npy')
# id_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar10_test.npy')
# id_labels = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar10_test_label.npy')
# id_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar10_res34_test.npy')
# id_labels = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/cifar10_res34_test_label.npy')
name = 'MultiProj'
# ood_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/inat_resnet34.npy')

# ood_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/svhn_resnet34_ci100.npy')
ood_features = np.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features/inat_resnet34.npy')
# for i in range(5):
    # print(i)
multiple_proj(id_features, id_labels, inat_loader, ood_features, name, device, n_components=17)
