import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def pad_or_truncate_lidar(lidar_data, max_points=12000):
    """
    Pad or truncate the LiDAR point cloud to a fixed size.
    Args:
        lidar_data: LiDAR point cloud as a NumPy array of shape (N, 4).
        max_points: Desired fixed size (maximum number of points).
    Returns:
        Fixed-size tensor of shape (max_points, 4).
    """
    num_points = lidar_data.shape[0]

    if num_points > max_points:
        # Truncate the array
        lidar_data = lidar_data[:max_points, :]
    elif num_points < max_points:
        # Pad the array with zeros
        padding = np.zeros((max_points - num_points, 4), dtype=lidar_data.dtype)
        lidar_data = np.vstack((lidar_data, padding))
    
    return torch.tensor(lidar_data, dtype=torch.float32)

class SequentialAlignmentDataset(Dataset):
    def __init__(self, csv_path, lidar_folder, camera_folder):
        # Load CSV data
        self.data = pd.read_csv(csv_path)
        num_entries = len(self.data)

        # Sort Camera and LiDAR files by filename (assumed chronological order)
        self.camera_files = sorted(
            [os.path.join(camera_folder, f) for f in os.listdir(camera_folder) if f.endswith('.png')]
        )[:num_entries]
        self.lidar_files = sorted(
            [os.path.join(lidar_folder, f) for f in os.listdir(lidar_folder) if f.endswith('.npy')]
        )[:num_entries]

        # Ensure the number of entries matches
        if len(self.camera_files) != num_entries or len(self.lidar_files) != num_entries:
            print(f"WARNING: Files truncated to match CSV rows ({num_entries}).")
            self.camera_files = self.camera_files[:num_entries]
            self.lidar_files = self.lidar_files[:num_entries]

        # Transformations for camera images
        self.camera_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images
            transforms.ToTensor(),         # Convert to tensor
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tabular data from CSV
        row = self.data.iloc[idx]

        # Convert tabular data to numeric, replacing non-numeric values with 0.0
        tabular_input = row[["X", "Y", "Z", "Speed", "Acceleration", "Yaw", "dist_left", "dist_right"]]
        tabular_input = pd.to_numeric(tabular_input, errors='coerce').fillna(0).values

        # Target output (Steering, Throttle, Brake)
        target_output = row[["Steering", "Throttle", "Brake"]]
        target_output = pd.to_numeric(target_output, errors='coerce').fillna(0).values

        # LiDAR data
        lidar_data = np.load(self.lidar_files[idx])
        lidar_data = pad_or_truncate_lidar(lidar_data, max_points=12000)  # Pad or truncate

        # Camera data
        camera_image = Image.open(self.camera_files[idx]).convert('RGB')
        camera_image = self.camera_transform(camera_image)

        return {
            "tabular": torch.tensor(tabular_input, dtype=torch.float32),
            "lidar": lidar_data,
            "camera": camera_image,
            "target": torch.tensor(target_output, dtype=torch.float32),
        }

# Paths to your data
csv_path = r"C:\softwares\CARLA_0.9.15_nonBuild\DL_urban_24\slow_combine\slow_innerCircle\2024-12-13_18-25-40_vehicle_data.csv"
lidar_folder = r"C:\softwares\CARLA_0.9.15_nonBuild\DL_urban_24\slow_combine\slow_innerCircle\LiDAR"
camera_folder = r"C:\softwares\CARLA_0.9.15_nonBuild\DL_urban_24\slow_combine\slow_innerCircle\FrontCamera"

# Create dataset and DataLoader
dataset = SequentialAlignmentDataset(csv_path, lidar_folder, camera_folder)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Test the dataset
for batch in dataloader:
    print("Tabular Data:", batch["tabular"].shape)  # (batch_size, 8)
    print("LiDAR Data:", batch["lidar"].shape)      # (batch_size, 12000, 4)
    print("Camera Data:", batch["camera"].shape)    # (batch_size, 3, 128, 128)
    print("Target:", batch["target"].shape)         # (batch_size, 3)
    break
