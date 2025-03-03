import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
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

# Helper class for metric tracking
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
    
def validate(model, loader, criterion, device):
    """
    Validate the model on the validation dataset
    Args:
        model: The model to validate
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on
    Returns:
        tuple: (average loss, average mIoU)
    """
    model.eval()
    losses = AverageMeter()
    mious = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
                
            loss = criterion(outputs, targets)

            miou = calculate_miou(outputs, targets)
            if not np.isnan(miou):
                mious.update(miou)
            losses.update(loss.item())

    return losses.avg, mious.avg
    
def plot_training_metrics(train_losses, val_losses, train_mious, val_mious, 
                         total_time, best_metrics, save_dir, batch_size):
    """
    Plot and save training metrics
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_mious: List of training mIoU values
        val_mious: List of validation mIoU values
        total_time: Total training time in seconds
        best_metrics: Dictionary containing best metric values
        save_dir: Directory to save the plot
        batch_size: Batch size used in training
    """
    plt.figure(figsize=(12, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Top subplot for metrics
    ax1 = plt.subplot(gs[0])
    ax1.set_title(f'Feature Distillation Training Metrics - Batch Size {batch_size}', 
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
    plot_path = os.path.join(save_dir, 'feature_distillation_training_metrics.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

def train_with_distillation(teacher_model, student_model, train_loader, val_loader, 
                          num_epochs, device, save_dir, feature_loss_weight=0.1, 
                          final_feature_weight=0.5):
    """
    Training function with both intermediate and final feature distillation
    Args:
        teacher_model: The pretrained teacher model
        student_model: The student model to be trained
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        device: Device to train on
        save_dir: Directory to save checkpoints
        feature_loss_weight: Weight for intermediate feature distillation loss
        final_feature_weight: Weight for final output feature distillation loss
    """
    start_time = time.time()
    epoch_times = []
    
    # Get the dataset from the train_loader for class weights
    train_dataset = train_loader.dataset
    class_weights = calculate_class_weights(train_dataset)
    class_weights = class_weights.to(device)
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    cosine_loss = nn.CosineEmbeddingLoss()
    
    # Optimizer and scheduler
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Define feature extraction hooks
    teacher_features = {}
    student_features = {}
    
    def get_teacher_features(name):
        def hook(module, input, output):
            teacher_features[name] = output
        return hook
    
    def get_student_features(name):
        def hook(module, input, output):
            if isinstance(output, dict):
                student_features[name] = output['out']
            else:
                student_features[name] = output
        return hook
    
    # Register hooks for teacher model
    teacher_hooks = []
    teacher_hooks.append(teacher_model.backbone.layer1.register_forward_hook(
        get_teacher_features('layer1')))
    teacher_hooks.append(teacher_model.backbone.layer2.register_forward_hook(
        get_teacher_features('layer2')))
    teacher_hooks.append(teacher_model.backbone.layer3.register_forward_hook(
        get_teacher_features('layer3')))
    teacher_hooks.append(teacher_model.backbone.layer4.register_forward_hook(
        get_teacher_features('layer4')))
    
    # Register hooks for student model's backbone
    student_hooks = []
    student_hooks.append(student_model.backbone.register_forward_hook(
        get_student_features('backbone')))
    
    # Channel aligners for feature matching
    channel_aligners = {}
    
    def get_channel_aligner(in_channels, out_channels):
        if (in_channels, out_channels) not in channel_aligners:
            aligner = nn.Conv2d(in_channels, out_channels, 1).to(device)
            nn.init.kaiming_normal_(aligner.weight)
            channel_aligners[(in_channels, out_channels)] = aligner
        return channel_aligners[(in_channels, out_channels)]
    
    # Training metrics tracking
    best_metrics = {
        'epoch': 0,
        'train_loss': float('inf'),
        'val_loss': float('inf'),
        'train_miou': 0,
        'val_miou': 0
    }
    
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training metrics
        losses = AverageMeter()
        ce_losses = AverageMeter()
        intermediate_feature_losses = AverageMeter()
        final_feature_losses = AverageMeter()
        mious = AverageMeter()
        
        student_model.train()
        teacher_model.eval()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass with teacher (for feature extraction only)
            with torch.no_grad():
                teacher_output = teacher_model(images)
                if isinstance(teacher_output, dict):
                    teacher_final_features = teacher_output['out']
                else:
                    teacher_final_features = teacher_output
            
            # Forward pass with student
            student_output = student_model(images)
            if isinstance(student_output, dict):
                student_logits = student_output['out']
            else:
                student_logits = student_output
            
            # Calculate CE loss
            ce_loss_val = ce_loss(student_logits, targets)
            
            # Calculate intermediate feature distillation loss
            intermediate_feature_loss = 0
            student_feat = student_features.get('backbone', None)
            
            if student_feat is not None:
                for teacher_name, teacher_feat in teacher_features.items():
                    # Spatial dimension alignment
                    if student_feat.shape[2:] != teacher_feat.shape[2:]:
                        teacher_feat = F.interpolate(
                            teacher_feat,
                            size=student_feat.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Channel dimension alignment
                    if student_feat.shape[1] != teacher_feat.shape[1]:
                        aligner = get_channel_aligner(student_feat.shape[1], teacher_feat.shape[1])
                        student_feat = aligner(student_feat)
                    
                    # Calculate cosine similarity loss
                    intermediate_feature_loss += cosine_loss(
                        student_feat.view(student_feat.size(0), -1),
                        teacher_feat.view(teacher_feat.size(0), -1),
                        torch.ones(images.size(0)).to(device)
                    )
            
            # Calculate final output feature distillation loss
            if student_logits.shape[2:] != teacher_final_features.shape[2:]:
                teacher_final_features = F.interpolate(
                    teacher_final_features,
                    size=student_logits.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            final_feature_loss = cosine_loss(
                student_logits.view(student_logits.size(0), -1),
                teacher_final_features.view(teacher_final_features.size(0), -1),
                torch.ones(images.size(0)).to(device)
            )
            
            # Combined loss with weights
            total_loss = (ce_loss_val + 
                         feature_loss_weight * intermediate_feature_loss +
                         final_feature_weight * final_feature_loss)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            losses.update(total_loss.item())
            ce_losses.update(ce_loss_val.item())
            intermediate_feature_losses.update(intermediate_feature_loss.item())
            final_feature_losses.update(final_feature_loss.item())
            
            # Calculate and update mIoU
            miou = calculate_miou(student_logits.detach(), targets)
            if not np.isnan(miou):
                mious.update(miou)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'ce_loss': f'{ce_losses.avg:.4f}',
                'int_feat_loss': f'{intermediate_feature_losses.avg:.4f}',
                'final_feat_loss': f'{final_feature_losses.avg:.4f}',
                'mIoU': f'{mious.avg:.4f}'
            })
        
        train_losses.append(losses.avg)
        train_mious.append(mious.avg)
        
        # Validation
        val_loss, val_miou = validate(student_model, val_loader, ce_loss, device)
        val_losses.append(val_loss)
        val_mious.append(val_miou)
        
        # Calculate epoch time and update metrics
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update best metrics and save model if improved
        if val_miou > best_metrics['val_miou']:
            best_metrics.update({
                'epoch': epoch + 1,
                'train_loss': losses.avg,
                'val_loss': val_loss,
                'train_miou': mious.avg,
                'val_miou': val_miou
            })
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_miou': val_miou,
                'best_metrics': best_metrics,
                'channel_aligners': {k: v.state_dict() for k, v in channel_aligners.items()}
            }, os.path.join(save_dir, 'best_feature_model.pth'))
        
        # Print epoch summary
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Time for epoch: {epoch_time/60:.2f} minutes')
        print(f'Average epoch time: {np.mean(epoch_times)/60:.2f} minutes')
        print(f'Train Loss: {losses.avg:.4f}, Train mIoU: {mious.avg:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}')
        print(f'Best Val mIoU: {best_metrics["val_miou"]:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_miou': val_miou,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_mious': train_mious,
                'val_mious': val_mious,
                'best_metrics': best_metrics,
                'channel_aligners': {k: v.state_dict() for k, v in channel_aligners.items()}
            }, os.path.join(save_dir, f'distillation_checkpoint_epoch_{epoch+1}.pth'))
    
    # Remove hooks
    for hook in teacher_hooks + student_hooks:
        hook.remove()
    
    # Calculate total training time
    total_time = time.time() - start_time
    
    # Print final summary
    print('\nTraining Complete!')
    print(f'Total training time: {total_time/3600:.2f} hours')
    print(f'Average time per epoch: {np.mean(epoch_times)/60:.2f} minutes')
    print(f'Best validation mIoU: {best_metrics["val_miou"]:.4f} (Epoch {best_metrics["epoch"]})')
    
    # Plot and save training metrics
    plot_training_metrics(
        train_losses, val_losses, train_mious, val_mious,
        total_time,
        best_metrics,
        save_dir,
        train_loader.batch_size
    )
    
    return train_losses, val_losses, train_mious, val_mious
    
def main():
    # Training settings
    num_epochs = 50
    batch_size = 16
    feature_loss_weight = 0.1  # Weight for feature distillation loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model directory with timestamp
    save_dir = os.path.join('checkpoints', 'feature_distill_v4_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
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

    # Create models
    teacher_model = fcn_resnet50(pretrained=True)
    student_model = LightweightDeepLabV3()
    
    teacher_model.to(device)
    student_model.to(device)
    
    # Save training configuration
    config = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'feature_loss_weight': feature_loss_weight,
        'learning_rate': 1e-4,
        'teacher_model': 'fcn_resnet50',
        'student_model': 'LightweightDeepLabV3',
        'dataset': 'VOC2012',
        'image_size': 256,
        'device': str(device)
    }
    
    with open(os.path.join(save_dir, 'training_config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f'{key}: {value}\n')
    
    # Print training configuration
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("\n")
    
    try:
        # Train with feature distillation only
        train_metrics = train_with_distillation(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            save_dir=save_dir,
            feature_loss_weight=0.1,
            final_feature_weight=0.5
        )
        
        print("\nTraining completed successfully!")
        print(f"Model checkpoints and metrics saved in: {save_dir}")
        
        # Unpack training metrics
        train_losses, val_losses, train_mious, val_mious = train_metrics
        
        # Print final performance
        print("\nFinal Performance:")
        print(f"Train Loss: {train_losses[-1]:.4f}")
        print(f"Validation Loss: {val_losses[-1]:.4f}")
        print(f"Train mIoU: {train_mious[-1]:.4f}")
        print(f"Validation mIoU: {val_mious[-1]:.4f}")
        print(f"Best Validation mIoU: {max(val_mious):.4f}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    
        # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Run main training loop
    main()