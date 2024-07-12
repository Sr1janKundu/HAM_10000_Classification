'''
Imports
'''
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2


'''
Transformations
'''
data_transforms_album = {
    "train": A.Compose([
        A.Resize(224, 224),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "test": A.Compose([
        A.Resize(224, 224),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


'''
DataClass
'''
class HAM10000_BINARY(Dataset):
    def __init__(self, root_dir, metadata_df, transform = None):
        self.root_dir = root_dir
        self.metadata_df = metadata_df
        self.transform = transform

        self.all_image_paths = os.listdir(self.root_dir)
        self.labels = self.metadata_df.set_index('image_id')['dx_binary'].to_dict()
        self.image_names = list(self.labels.keys())

    def __len__(self):
        return(len(self.image_names))

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_names[idx]+'.jpg')
        image = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = self.labels[self.image_names[idx]]

        if self.transform:
            image = self.transform(image=image)["image"]        # Albumentations returns a dictionary with keys like 'image', 'mask', etc., depending on the transformations applied.

        return image, label
    

'''
DataLoader
'''

def get_loaders(root_data_path,
                metadata_path,
                dataset_class = HAM10000_BINARY, 
                train_trans = data_transforms_album['train'], 
                test_trans = data_transforms_album['test'], 
                batch = 32, 
                seed = None):
    
    metadata_all = pd.read_csv(metadata_path, low_memory=False)
    df_positive_1954 = metadata_all[metadata_all["dx_binary"] == 1].reset_index(drop=True).sample(1954)
    df_negative_1954 = metadata_all[metadata_all["dx_binary"] == 0].reset_index(drop=True).sample(1954)
    idx = [i for i in range(1954)]
    random.Random(seed).shuffle(idx)
    train_len, val_len = int(len(idx)*0.7), int(len(idx)*0.2)
    # test_len = len(idx) - train_len - val_len
    train_idx, val_idx, test_idx = idx[:train_len], idx[train_len:train_len+val_len], idx[train_len+val_len:]
    
    df_pos_train, df_pos_val, df_pos_test = df_positive_1954.iloc[train_idx], df_positive_1954.iloc[val_idx], df_positive_1954.iloc[test_idx]
    df_neg_train, df_neg_val, df_neg_test = df_negative_1954.iloc[train_idx], df_negative_1954.iloc[val_idx], df_negative_1954.iloc[test_idx]
    df_train = pd.concat([df_pos_train, df_neg_train]).sample(frac=1).reset_index()
    df_val = pd.concat([df_pos_val, df_neg_val]).sample(frac=1).reset_index()
    df_test = pd.concat([df_pos_test, df_neg_test]).sample(frac=1).reset_index()

    train_ds = dataset_class(root_dir=root_data_path, metadata_df=df_train, transform=train_trans)
    val_ds = dataset_class(root_dir=root_data_path, metadata_df=df_val, transform=train_trans)
    test_ds = dataset_class(root_dir=root_data_path, metadata_df=df_test, transform=test_trans)

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=True)
    test_dl = DataLoader(test_ds, shuffle=False)

    return train_dl, val_dl, test_dl
