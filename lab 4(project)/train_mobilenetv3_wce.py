import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large 


from student_model import LightweightDeepLabV3


def calculate_class_weights(dataset, num_classes=21):
    """
    Compute class weights based on class frequency in the dataset
    """
    class_counts = torch.zeros(num_classes)
    print("Computing class weights...")
    
    for _, target in tqdm(dataset):
        classes, counts = torch.unique(target, return_counts=True)
        for class_id, count in zip(classes, counts):
            if class_id < num_classes:  # Ignore 255 (background) class
                class_counts[class_id] += count
    
    # Add small epsilon to avoid division by zero
    class_counts = class_counts + 1e-6
    
    # Compute weights as inverse of frequency
    weights = 1.0 / class_counts
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    print("Class weights calculated:")
    for i, weight in enumerate(weights):
        print(f"Class {i}: {weight:.4f}")
    return weights

def calculate_miou(pred, target, num_classes=21):
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()

    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    mious = AverageMeter()

    epoch_start_time = time.time()
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        miou = calculate_miou(outputs.detach(), targets)
        if not np.isnan(miou):
            mious.update(miou)
        losses.update(loss.item())

        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'mIoU': f'{mious.avg:.4f}'
        })

    epoch_time = time.time() - epoch_start_time
    return losses.avg, mious.avg, epoch_time

def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    mious = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            miou = calculate_miou(outputs, targets)
            if not np.isnan(miou):
                mious.update(miou)
            losses.update(loss.item())

    return losses.avg, mious.avg

def plot_training_metrics(train_losses, val_losses, train_mious, val_mious, 
                         total_time, best_metrics, save_dir, batch_size):
    plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Top subplot for metrics
    ax1 = plt.subplot(gs[0])
    ax1.set_title(f'Mobilenet Training Metrics - Batch Size {batch_size}', 
                 fontsize=14, pad=20)
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    # Create second y-axis for mIoU
    ax2 = ax1.twinx()
    ax2.plot(train_mious, label='Train mIoU', color='cyan', linestyle='--')
    ax2.plot(val_mious, label='Validation mIoU', color='magenta', linestyle='--')
    ax2.set_ylabel('mIoU', color='magenta')
    ax2.tick_params(axis='y', labelcolor='magenta')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Bottom subplot for metrics text
    ax3 = plt.subplot(gs[1])
    ax3.axis('off')
    
    metrics_text = (
        f'Total Training Time: {total_time/3600:.2f} hours\n'
        f'Average Time per Epoch: {total_time/len(train_losses)/60:.2f} minutes\n'
        f'Best Epoch: {best_metrics["epoch"]}\n'
        f'Best Train Loss: {best_metrics["train_loss"]:.4f}\n'
        f'Best Validation Loss: {best_metrics["val_loss"]:.4f}\n'
        f'Best Train mIoU: {best_metrics["train_miou"]:.4f}\n'
        f'Best Validation mIoU: {best_metrics["val_miou"]:.4f}'
    )
    
    ax3.text(0.5, 0.5, metrics_text,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax3.transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'no_distill_training_metrics.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Training settings
    num_epochs = 50
    batch_size = 16
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model directory with timestamp
    save_dir = os.path.join('checkpoints',"no_distill_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long))
    ])

    train_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='train', download=True,
        transform=transform, target_transform=target_transform
    )

    val_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='val', download=True,
        transform=transform, target_transform=target_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = LightweightDeepLabV3()
    model.to(device)

    # Calculate class weights from training dataset
    class_weights = calculate_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop with timing and metric tracking
    best_miou = 0
    best_metrics = {'epoch': 0, 'train_loss': float('inf'), 'val_loss': float('inf'),
                   'train_miou': 0, 'val_miou': 0}
    
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    epoch_times = []
    
    total_start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        train_loss, train_miou, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        epoch_times.append(epoch_time)

        # Validate
        val_loss, val_miou = validate(model, val_loader, criterion, device)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mious.append(train_miou)
        val_mious.append(val_miou)

        # Update learning rate
        scheduler.step(val_loss)

        # Update best metrics
        if val_miou > best_miou:
            best_miou = val_miou
            best_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_miou': train_miou,
                'val_miou': val_miou
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
            }, os.path.join(save_dir, 'best_model.pth'))

        # Print epoch summary with timing information
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Time for epoch: {epoch_time/60:.2f} minutes')
        print(f'Average epoch time: {np.mean(epoch_times)/60:.2f} minutes')
        print(f'Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}')
        print(f'Best Val mIoU: {best_miou:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')

        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_miou': val_miou,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Calculate total training time
    total_time = time.time() - total_start_time
    
    # Create and save the final plot
    plot_training_metrics(
        train_losses, val_losses, train_mious, val_mious,
        total_time, best_metrics, save_dir, batch_size
    )

    # Print final timing summary
    print('\nTraining Complete!')
    print(f'Total training time: {total_time/3600:.2f} hours')
    print(f'Average time per epoch: {np.mean(epoch_times)/60:.2f} minutes')
    print(f'Best validation mIoU: {best_miou:.4f} (Epoch {best_metrics["epoch"]})')

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    main()
