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

# Dataset Class
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

        self.camera_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        tabular_input = row[["X", "Y", "Z", "Speed", "Acceleration", "Yaw", "dist_left", "dist_right"]].values
        tabular_input = np.array(tabular_input, dtype=np.float32)  # Convert to float32

        target_output = row[["Steering", "Throttle", "Brake"]].values
        target_output = np.array(target_output, dtype=np.float32)  # Convert to float32

        lidar_data = np.load(self.lidar_files[idx])[:, 2].astype(np.float32)
        lidar_data = torch.tensor(lidar_data).view(1, 100, -1)  # Reshape to (1, H, W)

        camera_image = Image.open(self.camera_files[idx]).convert('RGB')
        camera_image = self.camera_transform(camera_image)

        return {
            "tabular": torch.tensor(tabular_input, dtype=torch.float32),
            "lidar": lidar_data,
            "camera": camera_image,
            "target": torch.tensor(target_output, dtype=torch.float32),
        }

# Model Definition
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

# Training Function
def train(model, train_loader, val_loader, loss_function, optimizer, num_epochs=100, device="cuda", save_dir="./results"):
    model.to(device)
    scaler = GradScaler()
    train_losses, val_losses = [], []

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            tabular_input = batch["tabular"].to(device)
            lidar_input = batch["lidar"].to(device)
            camera_input = batch["camera"].to(device)
            target_output = batch["target"].to(device)

            optimizer.zero_grad()
            with autocast():
                steering, brake, throttle = model(tabular_input, lidar_input, camera_input)
                outputs = torch.cat((steering, brake, throttle), dim=1)
                loss = loss_function(outputs, target_output)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tabular_input = batch["tabular"].to(device)
                lidar_input = batch["lidar"].to(device)
                camera_input = batch["camera"].to(device)
                target_output = batch["target"].to(device)

                steering, brake, throttle = model(tabular_input, lidar_input, camera_input)
                outputs = torch.cat((steering, brake, throttle), dim=1)
                loss = loss_function(outputs, target_output)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save model every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))

    # Final loss curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()

# Main Script
if __name__ == "__main__":
    # Paths
    csv_path = r"YOUR_CSV_PATH.csv"
    lidar_folder = r"YOUR_LIDAR_PATH"
    camera_folder = r"YOUR_CAMERA_PATH"
    save_dir = r"YOUR_SAVE_DIR"

    # Dataset
    dataset = SequentialAlignmentDataset(csv_path, lidar_folder, camera_folder)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    model = MultiModalNet()
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train
    train(model, train_loader, val_loader, loss_function, optimizer, num_epochs=20, save_dir=save_dir)

    # Test
    model.load_state_dict(torch.load(os.path.join(save_dir, "model_epoch_100.pth")))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            tabular_input = batch["tabular"].to("cuda")
            lidar_input = batch["lidar"].to("cuda")
            camera_input = batch["camera"].to("cuda")
            target_output = batch["target"].to("cuda")

            steering, brake, throttle = model(tabular_input, lidar_input, camera_input)
            outputs = torch.cat((steering, brake, throttle), dim=1)
            loss = loss_function(outputs, target_output)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
