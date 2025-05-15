import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image

from torchvision.datasets import VisionDataset, ImageFolder

from dataset import ImageNetValDataset
from transforms import fog, plasma_fractal


class CorruptedDataset(VisionDataset):
    def __init__(self,
        root: str = 'data',
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        corruption: str = "gaussian_noise",
        severity: int = 1,
        dataset: str = "c10"
        ) -> None:

        self.root = root
        self.train = train
        self.dataset = dataset
        self.corruption = corruption 

        if dataset.lower() == 'imagenet':
            self.use_imagenet = True
            # here only set for fog  
            self.corruption_func = fog
            self.severity = severity
            
            folder = 'train' if train else 'val'
            imagenet_root = "/oscar/data/shared/imagenet/ILSVRC2012"
            self.data_folder = os.path.join(imagenet_root, folder)

            if train:
                 train_dir = os.path.join(imagenet_root, "train")
                 if not os.path.exists(train_dir):
                    raise FileNotFoundError(f"ImageNet training directory not found at {train_dir}")
                
                 print(f"Using ImageNet training set from {train_dir} with {corruption} corruption (severity {severity})")
                 self.image_dataset = ImageFolder(train_dir)
            else:
                val_dir = os.path.join(imagenet_root, "val")
                val_gt_path = os.path.join(imagenet_root, "devkit/data/ILSVRC2012_validation_ground_truth.txt")
                meta_mat_path = os.path.join(imagenet_root, "devkit/data/meta.mat")
                
                print(f"Using ImageNet validation set from {val_dir} with {corruption} corruption (severity {severity})")
                self.image_dataset = ImageNetValDataset(
                    val_dir=val_dir,
                    val_gt_path=val_gt_path,
                    meta_mat_path=meta_mat_path,
                    transform=None  
                )
        else:     
            self.use_imagenet = False
            dataset_str = "_train" if train else "_test"
            # load data and labels
            self.data = np.float32(np.load(os.path.join(root, dataset + '-C', corruption + f'_{severity}' + dataset_str + '.npy')))
            self.targets = np.load(os.path.join(root, dataset + '-C', 'labels' + dataset_str + '.npy')).astype(int)
            
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if hasattr(self, 'use_imagenet') and self.use_imagenet:
            img, target = self.image_dataset[index]
            
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))
                
            if self.corruption == "fog":

                img = img.resize((224, 224))

                c = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][self.severity - 1]
                x = np.array(img) / 255.
                max_val = x.max()
                
                start = (256 - 224) // 2
                x += c[0] * plasma_fractal(mapsize=256, wibbledecay=c[1])[start:start+224, start:start+224][..., np.newaxis]
                
                corrupted_img = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
                corrupted_img = Image.fromarray(np.uint8(corrupted_img))
            else:
                corrupted_img = self.corruption_func(img, self.severity)
                
            # convert back to PIL Image if needed
            if not isinstance(corrupted_img, Image.Image):
                corrupted_img = Image.fromarray(np.uint8(corrupted_img))
                
            if self.transform is not None:
                corrupted_img = self.transform(corrupted_img)
                
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            return corrupted_img, target
        else:
            img, target = self.data[index], self.targets[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.fromarray(img.astype(np.uint8))
            
            if self.transform is not None:
                img = self.transform(img)
                
            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target

    def __len__(self) -> int:
        if hasattr(self, 'image_dataset'):
            return len(self.image_dataset)
        return len(self.data)
    