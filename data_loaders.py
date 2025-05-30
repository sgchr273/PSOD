import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import SVHN

# Shared CIFAR transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010]),
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2023, 0.1994, 0.2010]),
])

# Shared ImageNet‐scale transforms
transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# DataLoader kwargs
DL_KWARGS = {"num_workers": 2, "pin_memory": True}


def id_dataloaders(id_data: str, base_root: str, batch_size: int = 1000):
    """
    Returns test_loader for CIFAR-10/100 and Imagenet1k. We do not return train_loader since model 
    is already trained and inference is performed using test_loader.
    """
    idd = id_data.lower()
    if idd in ("cifar-10", "cifar10"):
        path = os.path.join(base_root, "CIFAR10", "id_data")
        ds_train = torchvision.datasets.CIFAR10(path, train=True,  download=True, transform=train_transform)
        ds_test  = torchvision.datasets.CIFAR10(path, train=False, download=True, transform=test_transform)
        DataLoader(ds_train, batch_size=batch_size, shuffle=True,  **DL_KWARGS)
        return DataLoader(ds_test, batch_size=batch_size, shuffle=False, **DL_KWARGS)

    if idd in ("cifar-100", "cifar100"):
        path = os.path.join(base_root, "CIFAR100")
        ds_train = torchvision.datasets.CIFAR100(path, train=True,  download=True, transform=train_transform)
        ds_test  = torchvision.datasets.CIFAR100(path, train=False, download=True, transform=test_transform)
        DataLoader(ds_train, batch_size=batch_size, shuffle=True,  **DL_KWARGS)
        return DataLoader(ds_test, batch_size=batch_size, shuffle=False, **DL_KWARGS)

    if idd in ("imagenet-1k", "imagenet1k"):
        # train_dir = os.path.join(base_root, "Imagenet", "id_data", "train")
        val_dir   = os.path.join(base_root, "Imagenet", "id_data", "imagenet-val")
        # ds_train = torchvision.datasets.ImageFolder(train_dir, transform=transform_test_largescale)
        ds_val   = torchvision.datasets.ImageFolder(val_dir,   transform=transform_test_largescale)
        # DataLoader(ds_train, batch_size=600, shuffle=True, **DL_KWARGS)
        return DataLoader(ds_val,   batch_size=1000, shuffle=False, **DL_KWARGS)

    raise ValueError(f"Unsupported id_data: {id_data}")


def ood_dataloaders(ood_name: str,
                    id_data: str,
                    base_root: str,
                    batch_size: int = 1000):
    """
    Return a DataLoader for the requested OOD dataset,
    using test_transform for CIFAR‐ID and transform_test_largescale for ImageNet‐ID.
    """
    # o = ood_name.lower()
    idd = id_data.lower()

    # pick appropriate transform based on id_data
    if idd in ("cifar-10", "cifar10", "cifar_10", "cifar10", "cifar-100", "cifar100", "cifar_100", "cifar100"):
        transform = test_transform
    elif idd in ("imagenet-1k", "imagenet1k"):
        transform = transform_test_largescale
    else:
        raise ValueError(f"Unsupported id_data: '{id_data}'")

    # ─── Imagenet‐style OOD ( iNaturalist, Places, SUN, Textures) ─────────
    if ood_name in ("iNaturalist", "Places", "SUN", "Textures"):
        sub = "Textures/images" if ood_name == "Textures" else ood_name
        path = os.path.join(base_root, "Imagenet", "ood_data", sub)
        ds = torchvision.datasets.ImageFolder(path, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, **DL_KWARGS)

    # ─── SVHN via torchvision (only for CIFAR‐ID) ───────────────────────────────
    if ood_name == "SVHN":
        svhn_root = os.path.join(base_root, "CIFAR10", "ood_data", "SVHN")
        ds = SVHN(root=svhn_root, split="test", download=True, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, **DL_KWARGS)

    # ─── LSUN & iSUN under CIFAR10/ood_data ────────────────────────────────────
    if ood_name in ("LSUN", "iSUN"):
        path = os.path.join(base_root, "CIFAR10", "ood_data", ood_name)
        ds = torchvision.datasets.ImageFolder(path, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, **DL_KWARGS)

    raise ValueError(f"Unsupported OOD dataset: '{ood_name}'")


