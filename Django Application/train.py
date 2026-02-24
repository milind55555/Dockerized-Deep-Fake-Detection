import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = "path/to/your/video/dataset"  # Should contain 'real/' and 'fake/' subdirs
BATCH_SIZE = 4
NUM_FRAMES = 16      # Number of frames per clip
IMG_SIZE = 112       # Resize frame to this size
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Video Dataset Class
# ----------------------------
class DeepfakeVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=16, img_size=112):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.img_size = img_size
        self.samples = []
        for label, subdir in enumerate(['fake', 'real']):
            folder = os.path.join(root_dir, subdir)
            if not os.path.isdir(folder):
                continue
            for video_file in os.listdir(folder):
                if video_file.endswith(('.mp4', '.avi')):
                    self.samples.append((os.path.join(folder, video_file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)
        if self.transform:
            frames = self.transform(frames)
        return frames, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < self.num_frames:
            # Pad by repeating last frame
            indices = list(range(frame_count)) + [frame_count - 1] * (self.num_frames - frame_count)
        else:
            # Uniformly sample frames
            step = frame_count // self.num_frames
            indices = [i * step for i in range(self.num_frames)]

        frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frames.append(frame)
        cap.release()

        # Handle case where too few frames were read
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))

        frames = np.array(frames[:self.num_frames])  # Shape: (T, H, W, C)
        frames = frames.transpose(3, 0, 1, 2)        # Shape: (C, T, H, W)
        frames = frames.astype(np.float32) / 255.0
        return torch.tensor(frames)

# ----------------------------
# 3D CNN Model
# ----------------------------
class DeepfakeDetector3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------
# Training Loop
# ----------------------------
def train():
    dataset = DeepfakeVideoDataset(
        root_dir=DATA_DIR,
        num_frames=NUM_FRAMES,
        img_size=IMG_SIZE
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = DeepfakeDetector3D().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for videos, labels in pbar:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100.*correct/total:.2f}%"
            })

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(dataloader):.4f}, Acc: {100.*correct/total:.2f}%")

    torch.save(model.state_dict(), "model_87_acc_20_frames_final_data.pt")
    print("Model saved as model_87_acc_20_frames_final_data.pt")

if __name__ == "__main__":
    train()