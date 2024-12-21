import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# Function to pad or truncate LiDAR data
def pad_or_truncate_lidar(lidar_data, max_points=12000):
    num_points = lidar_data.shape[0]
    if num_points > max_points:
        lidar_data = lidar_data[:max_points, :]
    elif num_points < max_points:
        padding = np.zeros((max_points - num_points, 4), dtype=lidar_data.dtype)
        lidar_data = np.vstack((lidar_data, padding))
    
    lidar_data = lidar_data[:, 2].reshape(1, 100, 120)  # Use Z values
    return torch.tensor(lidar_data, dtype=torch.float32)

# Dataset class
class SequentialAlignmentDataset(Dataset):
    def __init__(self, csv_path, lidar_folder, camera_folder):
        self.data = pd.read_csv(csv_path)
        num_entries = len(self.data)

        self.camera_files = sorted(
            [os.path.join(camera_folder, f) for f in os.listdir(camera_folder) if f.endswith('.png')]
        )[:num_entries]
        self.lidar_files = sorted(
            [os.path.join(lidar_folder, f) for f in os.listdir(lidar_folder) if f.endswith('.npy')]
        )[:num_entries]

        # Transform for camera images
        self.camera_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        tabular_input = row[["X", "Y", "Z", "Speed", "Acceleration", "Yaw", "dist_left", "dist_right"]]
        tabular_input = pd.to_numeric(tabular_input, errors='coerce').fillna(0).values

        target_output = row[["Steering", "Throttle", "Brake"]]
        target_output = pd.to_numeric(target_output, errors='coerce').fillna(0).values

        lidar_data = np.load(self.lidar_files[idx])
        lidar_data = pad_or_truncate_lidar(lidar_data)

        camera_image = Image.open(self.camera_files[idx]).convert('RGB')
        camera_image = self.camera_transform(camera_image)

        return {
            "tabular": torch.tensor(tabular_input, dtype=torch.float32),
            "lidar": lidar_data,
            "camera": camera_image,
            "target": torch.tensor(target_output, dtype=torch.float32),
        }

# Model definition
class MultiModalNet(nn.Module):
    def __init__(self):
        super(MultiModalNet, self).__init__()

        self.cnn_rgb = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.rgb_out_dim = 128 * 16 * 16

        self.cnn_lidar = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.lidar_out_dim = 128 * 13 * 15

        self.fc_tabular = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.fusion_fc = nn.Sequential(
            nn.Linear(57856, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.fc_steering = nn.Linear(256, 1)
        self.fc_brake = nn.Linear(256, 1)
        self.fc_throttle = nn.Linear(256, 1)

    def forward(self, tabular_input, lidar_input, camera_input):
        rgb_features = self.cnn_rgb(camera_input)
        lidar_features = self.cnn_lidar(lidar_input)
        tabular_features = self.fc_tabular(tabular_input)

        fused_features = torch.cat((rgb_features, lidar_features, tabular_features), dim=1)
        fused_output = self.fusion_fc(fused_features)

        steering = self.fc_steering(fused_output)
        brake = self.fc_brake(fused_output)
        throttle = self.fc_throttle(fused_output)

        return steering, brake, throttle

# Validation/Test function
def evaluate(model, dataloader, loss_function, device="cuda"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            tabular = batch["tabular"].to(device)
            lidar = batch["lidar"].to(device)
            camera = batch["camera"].to(device)
            target = batch["target"].to(device)

            steering, brake, throttle = model(tabular, lidar, camera)
            outputs = torch.cat((steering, brake, throttle), dim=1)
            loss = loss_function(outputs, target)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Training function
def train(model, train_loader, val_loader, test_loader, loss_function, optimizer, num_epochs, save_dir, device="cuda"):
    model.to(device)
    scaler = GradScaler()
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            tabular = batch["tabular"].to(device)
            lidar = batch["lidar"].to(device)
            camera = batch["camera"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            with autocast():
                steering, brake, throttle = model(tabular, lidar, camera)
                outputs = torch.cat((steering, brake, throttle), dim=1)
                loss = loss_function(outputs, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_function, device)

        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("Best model saved!")

        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
            print(f"Model checkpoint saved at epoch {epoch}.")

    # Evaluate test loss after training
    test_loss = evaluate(model, test_loader, loss_function, device)
    print(f"Final Test Loss: {test_loss:.4f}")

# Main script
if __name__ == "__main__":
    csv_path = r"C:\softwares\CARLA_0.9.15_nonBuild\DL_urban_24\slow_combine\slow_innerCircle\2024-12-13_18-25-40_vehicle_data.csv"
    lidar_folder = r"C:\softwares\CARLA_0.9.15_nonBuild\DL_urban_24\slow_combine\slow_innerCircle\LiDAR"
    camera_folder = r"C:\softwares\CARLA_0.9.15_nonBuild\DL_urban_24\slow_combine\slow_innerCircle\FrontCamera"
    save_dir = r"C:\softwares\CARLA_0.9.15_nonBuild\DL_urban_24\slow_combine\slow_innerCircle\Trainning_results"

    os.makedirs(save_dir, exist_ok=True)

    dataset = SequentialAlignmentDataset(csv_path, lidar_folder, camera_folder)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = MultiModalNet()
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    train(model, train_loader, val_loader, test_loader, loss_function, optimizer, num_epochs=100, save_dir=save_dir)
