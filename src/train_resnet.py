import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# --- 1. The Dataset Loader ---
class SensorDataset(Dataset):
    """Takes the raw long-format Parquet and reshapes it to (Channels, Time_Steps)"""
    def __init__(self, df_data, df_labels):
        self.labels = df_labels['class_label'].values - 1 # 0-5
        self.sample_ids = df_labels['Sample_ID'].values
        self.data_grouped = df_data.groupby('Sample_ID')
        self.signal_cols = [c for c in df_data.columns if 'signal' in c.lower()]
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        s_id = self.sample_ids[idx]
        # Get the 100 rows for this sample
        sample_data = self.data_grouped.get_group(s_id)[self.signal_cols].values
        # PyTorch expects (Channels, Sequence_Length), so we transpose (14, 100)
        tensor_data = torch.tensor(sample_data.T, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_data, label

# --- 2. The 1D ResNet Architecture ---
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection adjustment if channels change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(res) # The magic of ResNet!
        x = self.relu(x)
        return x

class SensorResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Input: 14 sensors. Output: 64 features.
        self.initial_layer = nn.Sequential(
            nn.Conv1d(14, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Stacking Residual Blocks
        self.layer1 = ResidualBlock1D(64, 64)
        self.layer2 = ResidualBlock1D(64, 128)
        self.layer3 = ResidualBlock1D(128, 256)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.initial_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).squeeze(-1) # Flatten
        x = self.fc(x)
        return x

# --- 3. Training Loop ---
def main():
    print("=== Phase 2: Training 1D ResNet on Augmented Data ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the combined data from Phase 1
    df_train1 = pd.read_parquet("data/processed/train_imputed.parquet")
    try:
        df_train2 = pd.read_parquet("data/processed/pseudo_imputed.parquet")
        df_data = pd.concat([df_train1, df_train2], ignore_index=True)
    except FileNotFoundError:
        df_data = df_train1 # Fallback if pseudo didn't generate
        
    df_labels = pd.read_csv("data/processed/combined_labels.csv")
    
    # Initialize PyTorch Dataset
    dataset = SensorDataset(df_data, df_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize Model, Loss, and Optimizer
    model = SensorResNet(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    epochs = 15
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(dataloader):.4f} - Train Macro-F1: {epoch_f1:.4f}")
        torch.save(model.state_dict(), "data/processed/resnet_weights.pth")
        print("Model saved successfully!")
if __name__ == "__main__":
    main()