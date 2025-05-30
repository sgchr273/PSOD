import torch
import torch.nn as nn
from torchvision import models
import argparse
import utils_ood as utils
from resnet import ResNet34, ResNet18
from data_loaders import id_dataloaders, ood_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi‐projection OOD experiment on pre‐extracted features."
    )
    parser.add_argument(
        "--id_data",
        choices=["cifar10", "cifar100", "imagenet1k"],
        required=True,
        help="In-distribution dataset."
    )
    parser.add_argument(
        "--ood_data",
        required=True,
        help=(
            "Out-of-distribution dataset. Valid values depend on --id_data:\n"
            "  imagenet-1k → DTD, iNaturalist, Places, SUN, Textures\n"
            "  cifar_100|cifar_10 → iSUN, LSUN, SVHN, Places, Textures, iNaturalist"
        )
    )
    parser.add_argument(
        "--model_name",
        choices=["resnet18", "resnet34", "resnet50", "mobilenet"],
        required=True,
        help="Feature extractor architecture."
    )
    return parser.parse_args()




def get_num_classes(id_data: str) -> int:
    if id_data == 'cifar10':
        return 10
    elif id_data == 'cifar100':
        return 100
    else:
        return 1000


def get_num_components(id_data: str, model_name: str) -> int:
    if id_data == "cifar10":
        return 100
    elif id_data == "cifar100":
        return 20
    else:  # imagenet-1k
        return 17 if model_name == "resnet50" else 20


def get_model(model_name: str, num_classes: int):
    if model_name == 'resnet18':
        model = ResNet18(num_class=num_classes)
    elif model_name == 'resnet34':
        model = ResNet34(num_class=num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # mobilenet
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def extract_features(model, loader, model_name: str):
    model = model.to(device).eval()

    if model_name == "resnet50":
        penult = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    elif model_name == "mobilenet":
        penult = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)).to(device).eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if model_name == "resnet18" or model_name == "resnet34":
                _, _, feat, _ = model(images)
            else:
               feat = penult(images).squeeze(-1).squeeze(-1)
            features.append(feat.cpu())
            labels.append(targets.cpu())
    print("Extracted features:", len(features), "batches")
    return torch.cat(features), torch.cat(labels)


def create_subspaces(features, labels, n_components):
    """
    Build class-specific subspaces by SVD on the raw feature matrices.
    - Moves class features to `device` before SVD for GPU acceleration.
    - Uses scalar Python ints for class keys.
    """
    subspaces = {}
    for cls in torch.unique(labels):
        k = cls.item()  # convert 0-d tensor to Python int
        # select and move to device
        cls_feats = features[labels == cls].to(device)  # [N_k, D] on correct device
        
        # compute SVD
        U, S, Vh = torch.linalg.svd(cls_feats, full_matrices=False)  # Vh: [D, D]
        
        # take at most n_components directions
        num_dirs = min(n_components, Vh.size(0))
        subspaces[k] = Vh[:num_dirs].to(device)  # ensure stored on device

    print("Created subspaces for", len(subspaces), "classes")
    return subspaces



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

def main():
    args = parse_args()
    BASE = "/usr/local/home/sgchr/Documents/OOD/Multiple_spaces"
    recall = 0.95
    print(f"Running experiment with ID dataset = {args.id_data}, "
        f"OOD dataset = {args.ood_data}, "
        f"Model = {args.model_name}")

    # 1) Model & components
    num_classes  = get_num_classes(args.id_data)
    n_components = get_num_components(args.id_data, args.model_name)
    model        = get_model(args.model_name, num_classes)

    ### Loading trained model's checkpoint that gives ~93% test accuracy on ID dataset. For 
    ### imagenet-1k we use the pretrained checkpoints of resnet50 and mobilenet. 
    if args.id_data in ['cifar10','cifar100']:
        model.load_state_dict(torch.load(f'/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/model_checkpoints/{args.model_name}_{args.id_data}.pth'))

    # 2) DataLoaders
    val_loader = id_dataloaders(args.id_data, BASE)
    ood_loader = ood_dataloaders(args.ood_data, args.id_data, BASE)
    # 3) Feature Extraction
    id_features, id_labels = extract_features(model, val_loader, args.model_name)
    ood_features, _  = extract_features(model, ood_loader, args.model_name)

    # 4) Subspace & metrics
    subspaces   = create_subspaces(id_features, id_labels, n_components)
    id_scores   = calculate_best_errors(id_features, subspaces)
    ood_scores  = calculate_best_errors(ood_features, subspaces)

    auc_ood     = utils.auc(ood_scores, id_scores)[0]
    fpr95, _    = utils.fpr_recall(ood_scores, id_scores, recall)

    print(f"\nResults on {args.id_data} vs {args.ood_data} using {args.model_name}:")
    print(f"  AUROC = {auc_ood:.4f}    FPR@95% = {fpr95:.4f}")


if __name__ == "__main__":
    main()


