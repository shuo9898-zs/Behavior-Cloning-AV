##implement_carla

import carla
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import random
import cv2
import os

# ==== Neural Network Model ====
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
        self.tabular_out_dim = 128

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

# ==== Preprocessing Functions ====
def preprocess_camera(image):
    """Preprocess camera image."""
    img_np = np.array(image.raw_data).reshape((720, 1280, 4))[:, :, :3]  # Assumes 1280x720 RGB
    img_resized = cv2.resize(img_np, (128, 128))  # Resize to model input size
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)  # Add batch dimension
    return img_tensor

def preprocess_lidar(lidar_data):
    """Preprocess LiDAR data."""
    lidar_np = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    max_points = 12000
    if len(lidar_np) > max_points:
        lidar_np = lidar_np[:max_points]
    elif len(lidar_np) < max_points:
        padding = np.zeros((max_points - len(lidar_np), 4))
        lidar_np = np.vstack((lidar_np, padding))
    height, width = 100, max_points // 100
    lidar_tensor = torch.tensor(lidar_np[:, 2].reshape(1, height, width), dtype=torch.float32)  # Use z-axis
    return lidar_tensor

def preprocess_tabular(location, velocity, acceleration, yaw, dist_left, dist_right):
    """Preprocess tabular data."""
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    accel = np.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
    tabular_data = [location.x, location.y, location.z, speed, accel, yaw, dist_left, dist_right]
    return torch.tensor(tabular_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# ==== Implement Model in CARLA ====
def implement_model_in_carla(model_path="trained_model.pth"):
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MultiModalNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    def spawn_vehicle_and_sensors():
        """Spawns a vehicle with sensors."""
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # Attach camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Attach LiDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        return vehicle, camera, lidar

    def process_data(camera, lidar, vehicle):
        """Processes input data and applies control to the vehicle."""
        camera_data = preprocess_camera(camera)
        lidar_data = preprocess_lidar(lidar)

        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        dist_left, dist_right = 0.0, 0.0  # Placeholder for lane distances
        tabular_data = preprocess_tabular(transform.location, velocity, acceleration, transform.rotation.yaw, dist_left, dist_right)

        # Get model predictions
        with torch.no_grad():
            steering, brake, throttle = model(tabular_data.to(device), lidar_data.to(device), camera_data.to(device))

        # Apply control to the vehicle
        control = carla.VehicleControl(
            steer=float(steering.item()),
            throttle=float(throttle.item()),
            brake=float(brake.item())
        )
        vehicle.apply_control(control)

    try:
        vehicle, camera, lidar = spawn_vehicle_and_sensors()
        print("Vehicle and sensors spawned. Press Ctrl+C to stop.")
        while True:
            process_data(camera.listen(lambda img: img), lidar.listen(lambda data: data), vehicle)

    except KeyboardInterrupt:
        print("Stopping the simulation...")
    finally:
        vehicle.destroy()
        camera.destroy()
        lidar.destroy()

# ==== Main ====
if __name__ == "__main__":
    implement_model_in_carla(model_path="best_model.pth")
