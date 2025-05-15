import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import urllib
import tarfile

from autoaugment import CIFAR10Policy, SVHNPolicy, ImageNetPolicy
from criterions import LabelSmoothingCrossEntropyLoss
from da import RandomCropPaste
from corrupted_datasets import CorruptedDataset
from dataset import ImageNetValDataset

def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):
    if args.model_name == 'vit':
        from vit import ViT
        net = ViT(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            attn=args.attn,
            init_values=args.init_values
            )
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net

def get_transform(args):
    train_transform = []
    test_transform = []

    # if args.train_corruption:
    #     train_transform += [CIFAR_Corruption(args.train_corruption), transforms.ToPILImage()]

    # if args.test_corruption:
    #     train_transform += [CIFAR_Corruption(args.test_corruption), transforms.ToPILImage()]
    
    if args.dataset == "imagenet" :
        train_transform += [transforms.RandomResizedCrop(224)]
        test_transform += [
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]

    if args.dataset == "imagenette":
        train_transform += [transforms.Resize((224, 224))]

    if args.dataset != "imagenet":    
        train_transform += [
            transforms.RandomCrop(size=args.size, padding=args.padding)
        ]
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]
    
    if args.autoaugment:
        if args.dataset == 'c10' or args.dataset=='c100':
            train_transform.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform.append(SVHNPolicy())
        elif args.dataset == "imagenette" or args.dataset == "imagenet":
            train_transform.append(ImageNetPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")   

    train_transform += [
        transforms.ToTensor()
    ]
    if not args.no_normalization:
        train_transform+=[transforms.Normalize(mean=args.mean, std=args.std)]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]
    
    if args.dataset == "imagenette":
        test_transform += [transforms.Resize((224, 224))]
    test_transform += [
        transforms.ToTensor(),
    ]
    if not args.no_normalization:
        test_transform += [transforms.Normalize(mean=args.mean, std=args.std)]
        #
        # test_transform += transforms.Normalize(mean=args.mean, std=args.std)

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        if args.train_corruption:
            train_ds = CorruptedDataset(root, train=True, transform=train_transform, corruption=args.train_corruption, severity=args.severity,dataset=args.dataset)
        else:
            train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)

        if args.test_corruption:
            test_ds = CorruptedDataset(root, train=False, transform=test_transform, corruption=args.test_corruption, severity=args.severity,dataset=args.dataset)
        else:
            test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        if args.train_corruption:
            train_ds = CorruptedDataset(root, train=True, transform=train_transform, corruption=args.train_corruption, severity=args.severity,dataset=args.dataset)
        else:
            train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        if args.test_corruption:
            test_ds = CorruptedDataset(root, train=False, transform=test_transform, corruption=args.test_corruption, severity=args.severity,dataset=args.dataset)
        else:
            test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "imagenette":
        os.makedirs(root, exist_ok=True)
    
        # URL for full-size Imagenette
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        filename = os.path.join(root, "imagenette2.tgz")
        
        # Download if not exists
        if not os.path.exists(filename):
            print(f"Downloading full-size Imagenette...")
            urllib.request.urlretrieve(url, filename)
            
        # Extract if not already extracted
        extract_path = os.path.join(root, "imagenette2")
        if not os.path.exists(extract_path):
            print("Extracting files...")
            with tarfile.open(filename, 'r:gz') as tar:
                tar.extractall(path=root)

        args.in_c = 3
        args.num_classes = 10
        args.size = 224
        args.padding = 28
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats
        train_transform, test_transform = get_transform(args)
        if args.train_corruption:
            train_ds = CorruptedDataset(root, train=True, transform=train_transform, corruption=args.train_corruption, severity=args.severity, dataset="imagenette")
        else:
            train_ds = torchvision.datasets.ImageFolder(
                os.path.join(extract_path, 'train'),
                transform=train_transform
            )
        if args.test_corruption:
            test_ds = CorruptedDataset(root, train=False, transform=test_transform, corruption=args.test_corruption, severity=args.severity, dataset="imagenette")
        else:
            test_ds = torchvision.datasets.ImageFolder(
                os.path.join(extract_path, 'val'),
                transform=test_transform
            )

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(root, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)

    # add imagenet
    elif args.dataset == "imagenet":

        if args.data_dir:
            imagenet_path = args.data_dir
        else:
            imagenet_path = "/oscar/data/shared/imagenet/ILSVRC2012"
            
        if not os.path.exists(imagenet_path):
            raise FileNotFoundError(f"ImageNet dataset not found at {imagenet_path}")
            
        print(f"Using ImageNet dataset from: {imagenet_path}")

        # Set parameters
        args.in_c = 3
        args.num_classes = 1000
        args.size = 224
        args.padding = 0
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        train_transform, test_transform = get_transform(args)

        # Use the train and val directories that already exist
        train_dir = os.path.join(imagenet_path, "train")
        val_dir = os.path.join(imagenet_path, "val")
        
        # Verify directories exist
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"ImageNet train directory not found at {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"ImageNet validation directory not found at {val_dir}")
        
        if args.train_corruption: 
            train_ds = CorruptedDataset(
                root=imagenet_path,
                train=True,
                transform=train_transform,
                corruption=args.train_corruption,
                severity=args.severity,
                dataset="imagenet"
            )
        else:
             train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        
        if args.test_corruption:
            test_ds = CorruptedDataset(
                root=imagenet_path,
                train=False,
                transform=test_transform,
                corruption=args.test_corruption,
                severity=args.severity,
                dataset="imagenet"
            )
        else:
            test_ds = ImageNetValDataset(
                val_dir=val_dir,
                val_gt_path=os.path.join(imagenet_path, "devkit/data/ILSVRC2012_validation_ground_truth.txt"),
                meta_mat_path=os.path.join(imagenet_path, "devkit/data/meta.mat"),
                transform=test_transform
            )
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    if args.attn_type:
        experiment_name+=f"_attn_{args.attn_type}"
    if args.train_corruption:
        experiment_name+=f"_train_{args.train_corruption}_{args.severity}"
    if args.test_corruption:
        experiment_name+=f"_test_{args.test_corruption}_{args.severity}"
    print(f"Experiment:{experiment_name}")
    return experiment_name
