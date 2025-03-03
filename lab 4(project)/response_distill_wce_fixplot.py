import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm
import numpy as np
from datetime import datetime
import time 
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
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

class AverageMeter:
    """Computes and stores the average and current value"""
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

def calculate_miou(pred, target, num_classes=21):
    """Calculate mean IoU for semantic segmentation"""
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

def train_knowledge_distillation(
    teacher,
    student,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    T,
    soft_target_loss_weight,
    ce_loss_weight,
    device,
    save_dir
):
    """
    Train student model using knowledge distillation from teacher model
    """
    teacher.eval()  # Teacher model is fixed during training
    student.train()
    
    # We need to get the dataset from the train_loader
    train_dataset = train_loader.dataset
    class_weights = calculate_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_miou = 0
    best_metrics = {
        'epoch': 0,
        'train_loss': float('inf'),
        'train_ce_loss': float('inf'),  # Track CE loss separately
        'val_loss': float('inf'),
        'train_miou': 0,
        'val_miou': 0
    }
    
    metrics = {
        'train_losses': [],
        'train_ce_losses': [],  # Track CE losses separately
        'val_losses': [],
        'train_mious': [],
        'val_mious': []
    }
    
    epoch_times = []
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training
        student.train()
        train_loss_meter = AverageMeter()  # Use AverageMeter for loss tracking
        train_ce_loss_meter = AverageMeter()  # Separate meter for CE loss
        train_miou = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_output = teacher(inputs)['out']
                
            # Get student predictions
            student_output = student(inputs)
            
            # Knowledge Distillation Loss
            soft_teacher = F.softmax(teacher_output / T, dim=1)
            soft_student = F.log_softmax(student_output / T, dim=1)
            distillation_loss = F.kl_div(
                soft_student,
                soft_teacher,
                reduction='batchmean'
            ) * (T * T)
            
            # Standard Cross-Entropy Loss
            ce_loss = criterion(student_output, labels)
            
            # Combined Loss
            loss = soft_target_loss_weight * distillation_loss + ce_loss_weight * ce_loss
            
            loss.backward()
            optimizer.step()
            
            # Update metrics using AverageMeter
            train_loss_meter.update(loss.item(), inputs.size(0))
            train_ce_loss_meter.update(ce_loss.item(), inputs.size(0))
            miou = calculate_miou(student_output.detach(), labels)
            if not np.isnan(miou):
                train_miou.update(miou)
            
            pbar.set_postfix({
                'total_loss': f'{train_loss_meter.avg:.4f}',
                'ce_loss': f'{train_ce_loss_meter.avg:.4f}',
                'mIoU': f'{train_miou.avg:.4f}'
            })
        
        # Get final training loss for the epoch
        train_loss = train_loss_meter.avg
        train_ce_loss = train_ce_loss_meter.avg
        
        # Validation
        val_loss, val_miou = validate_distillation(student, val_loader, criterion, device)
        
        # Update learning rate based on CE loss for better comparison
        scheduler.step(train_ce_loss)
        
        # Store metrics
        metrics['train_losses'].append(train_loss)
        metrics['train_ce_losses'].append(train_ce_loss)
        metrics['val_losses'].append(val_loss)
        metrics['train_mious'].append(train_miou.avg)
        metrics['val_mious'].append(val_miou)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Update best metrics and save best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_ce_loss': train_ce_loss,
                'val_loss': val_loss,
                'train_miou': train_miou.avg,
                'val_miou': val_miou
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou,
            }, os.path.join(save_dir, 'best_response_model.pth'))
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_miou': val_miou,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Print epoch summary
        print(f'\nEpoch {epoch + 1}/{epochs}:')
        print(f'Time for epoch: {epoch_time/60:.2f} minutes')
        print(f'Average epoch time: {np.mean(epoch_times)/60:.2f} minutes')
        print(f'Train Total Loss: {train_loss:.4f}, Train CE Loss: {train_ce_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Train mIoU: {train_miou.avg:.4f}')
        print(f'Val mIoU: {val_miou:.4f}, Best Val mIoU: {best_miou:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')
    
    # Calculate total training time
    total_time = time.time() - total_start_time
    
    # Create final plot with separate CE loss
    plot_distillation_metrics_with_ce(
        metrics['train_losses'],
        metrics['train_ce_losses'],
        metrics['val_losses'],
        metrics['train_mious'],
        metrics['val_mious'],
        total_time,
        best_metrics,
        save_dir,
        T,
        soft_target_weight=soft_target_loss_weight
    )
    
    # Print final timing summary
    print('\nTraining Complete!')
    print(f'Total training time: {total_time/3600:.2f} hours')
    print(f'Average time per epoch: {np.mean(epoch_times)/60:.2f} minutes')
    print(f'Best validation mIoU: {best_miou:.4f} (Epoch {best_metrics["epoch"]})')
    
    return metrics

def plot_distillation_metrics_with_ce(train_losses, train_ce_losses, val_losses, 
                                    train_mious, val_mious, total_time, best_metrics, 
                                    save_dir, temperature, soft_target_weight):
    """Plot training metrics including separate CE loss for better comparison"""
    plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Top subplot for metrics
    ax1 = plt.subplot(gs[0])
    ax1.set_title(f'Response Distillation Training Metrics\nT={temperature}, λ={soft_target_weight}', 
                 fontsize=14, pad=20)
    
    # Plot losses
    ax1.plot(train_ce_losses, label='Train CE Loss', color='blue')
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
        f'Best Train Total Loss: {best_metrics["train_loss"]:.4f}\n'
        f'Best Train CE Loss: {best_metrics["train_ce_loss"]:.4f}\n'
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
    plot_path = os.path.join(save_dir, 'response_distill_training_metrics.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

def validate_distillation(model, val_loader, criterion, device):
    """Validate the model during distillation training"""
    model.eval()
    val_loss = 0.0
    mious = AverageMeter()
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            miou = calculate_miou(outputs, labels)
            if not np.isnan(miou):
                mious.update(miou)
    
    return val_loss / len(val_loader), mious.avg
    


def plot_distillation_metrics(train_losses, val_losses, train_mious, val_mious,
                            total_time, best_metrics, save_dir, temperature, soft_target_weight):
    """Plot training metrics for knowledge distillation"""
    plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Top subplot for metrics
    ax1 = plt.subplot(gs[0])
    ax1.set_title(f'Response Distillation Training Metrics\nT={temperature}, λ={soft_target_weight}', 
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
    plot_path = os.path.join(save_dir, 'response_distill_loss.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
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

    # Setup datasets
    train_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='train', download=True,
        transform=transform, target_transform=target_transform
    )

    val_dataset = VOCSegmentation(
        root='./data', year='2012', image_set='val', download=True,
        transform=transform, target_transform=target_transform
    )

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # Create teacher and student models
    teacher = fcn_resnet50(pretrained=True).to(device)
    student = LightweightDeepLabV3().to(device)

    # Create model directory with timestamp
    save_dir = os.path.join('checkpoints', "response_distill_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    # Training parameters
    params = {
        'epochs': 50,
        'learning_rate': 1e-4,
        'T': 2.0,  # Temperature for softening probabilities
        'soft_target_loss_weight': 0.5,  # Weight for distillation loss
        'ce_loss_weight': 0.5,  # Weight for cross-entropy loss
        'save_dir': save_dir
    }

    # Train with knowledge distillation
    metrics = train_knowledge_distillation(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        **params
    )

if __name__ == "__main__":

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()