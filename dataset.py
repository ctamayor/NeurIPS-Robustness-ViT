import os
import scipy.io
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class ImageNetValDataset(Dataset):
    def __init__(self, val_dir, val_gt_path, meta_mat_path, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        
        # Get training class mapping directly from ImageFolder
        train_dir = os.path.join(os.path.dirname(val_dir), "train")
        dummy_train_ds = torchvision.datasets.ImageFolder(train_dir, transform=None)
        self.class_to_idx = dummy_train_ds.class_to_idx
        
        # Load mapping from validation label to WordNet ID
        meta = scipy.io.loadmat(meta_mat_path)
        synsets = meta["synsets"]
        self.index_to_wordnet = {int(entry[0][0][0]): str(entry[0][1][0]) for entry in synsets}
        
        # Load validation labels
        with open(val_gt_path, "r") as f:
            val_labels = [int(line.strip()) for line in f.readlines()]
        
        # Create list of (image_path, class_id)
        self.data = []
        for i, original_label in enumerate(val_labels):
            # Get WordNet ID for this validation label
            wordnet_id = self.index_to_wordnet.get(original_label, None)
            if wordnet_id and wordnet_id in self.class_to_idx:
                # Get the class index that ImageFolder would assign
                img_folder_idx = self.class_to_idx[wordnet_id]
                self.data.append((
                    os.path.join(val_dir, f"ILSVRC2012_val_{i+1:08d}.JPEG"),
                    img_folder_idx
                ))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label