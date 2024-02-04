import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

import dataset
import metric
import model

# Creating Dataset
train_dataset = dataset.SyntheticShapesDataset(500)
val_dataset = dataset.SyntheticShapesDataset(100)

# Creating Dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
model = model.UNet()

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Metric
iou_score = metric.iou_score

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training and Evaluation function
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_iou, train_dice = 0.0, 0.0

        # Training loop
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks/255)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += iou_score(outputs, masks/255).item()
            
        # Average over the dataset
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_iou, val_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks/255)
                val_loss += loss.item()
                val_iou += iou_score(outputs, masks/255).item()
                
        # Average over the dataset
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, IoU: {train_iou:.4f} - "
              f"Val loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

# Training
train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, num_epochs=15)

# Saving the model
torch.save(model, "unet_model.h5")
print("Model saved")