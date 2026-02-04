import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MedicalImageDataset(Dataset):
    def __init__(self, csv_path, image_root, image_size=256, transform=None):
        self.df = pd.read_csv(csv_path)
        #filter rows with missing images
        self.df = self.df[self.df.iloc[:, 0].apply(lambda rel_path: os.path.exists(os.path.join(image_root, rel_path)))]

        self.image_root = image_root
        self.image_size = image_size

        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),   # outputs [0,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.df.iloc[idx, 0]
        
        img_path = os.path.join(self.image_root, rel_path)

        image = Image.open(img_path).convert("RGB")  
        image = self.transform(image)

        return image, 0  # Return image and a dummy label (0) for compatibility
