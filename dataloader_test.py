
import torch
from torch.utils.data import DataLoader
from medical_dataset import MedicalImageDataset  # Assuming medical_dataset.py is in the same directory

# Initialize dataset (replace with your actual CSV and image root paths)
csv_path = './train_pairs.csv'  # e.g., 'data/labels.csv'
image_root = ''  # Current directory (CSV already has Images/ prefix in paths)

dataset = MedicalImageDataset(csv_path=csv_path, image_root=image_root)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test iteration (try to load a few batches)
try:
    for batch_idx, images in enumerate(dataloader):
        print(f"Batch {batch_idx}: Loaded {images.shape[0]} images with shape {images.shape}")
        if batch_idx >= 5:  # Test only first 3 batches
            break
    print("DataLoader test passed!")
except Exception as e:
    print(f"DataLoader test failed: {e}")