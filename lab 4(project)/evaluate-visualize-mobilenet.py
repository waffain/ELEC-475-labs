import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from student_model import LightweightDeepLabV3

# VOC class names for reference
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
    'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
    'train', 'tv/monitor'
]

# Define the color map for PASCAL VOC
label_colors = np.array([
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),  # potted plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128),  # tv/monitor
])



def create_color_mask(mask):
    """Convert segmentation mask to color image."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in enumerate(label_colors):
        color_mask[mask == label] = color
    return color_mask

def calculate_miou(pred, target, num_classes=21):
    """Calculate mean IoU and per-class IoUs."""
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
    return ious, np.nanmean(ious)

def visualize_prediction(image, target, prediction, title=None):
    """Visualize original image, ground truth, and prediction side by side."""
    # Convert tensors to numpy arrays
    image = image.squeeze(0).cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    target = target.squeeze(0).cpu().numpy()
    prediction = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    
    # Convert masks to color
    target_color = create_color_mask(target)
    pred_color = create_color_mask(prediction)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.05)
    
    # Plot images
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(target_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate semantic segmentation model')
    parser.add_argument('--weights', type=str, required=True,
                      help='Path to the model weights file')
    parser.add_argument('--data_root', type=str, default='./data',
                      help='Path to the dataset root directory')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained backbone')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = LightweightDeepLabV3(num_classes=21, pretrained=args.pretrained)
    
    # Load checkpoint
    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'val_miou' in checkpoint:
            print(f"Previous validation mIoU: {checkpoint['val_miou']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly")
    
    model.eval()
    model.to(device)

    # Define transforms
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

    # Load validation dataset
    dataset = VOCSegmentation(
        root=args.data_root,
        year='2012',
        image_set='val',
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Initialize storage for results
    all_data = []
    all_mious = []
    class_ious = [[] for _ in range(len(VOC_CLASSES))]
    total_miou = 0
    num_samples = 0
    
    best_miou = -float('inf')
    worst_miou = float('inf')
    best_idx = 0
    worst_idx = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(loader)):
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            output = model(images)
            
            all_data.append((images.cpu(), targets.cpu(), output.cpu()))
            
            ious, miou = calculate_miou(output, targets)
            
            if not np.isnan(miou):
                if miou > best_miou:
                    best_miou = miou
                    best_idx = idx
                if miou < worst_miou:
                    worst_miou = miou
                    worst_idx = idx
                
                all_mious.append(miou)
                total_miou += miou
                num_samples += 1
            
            for cls in range(len(VOC_CLASSES)):
                if not np.isnan(ious[cls]):
                    class_ious[cls].append(ious[cls])

    # Print evaluation results
    print("\nPer-class mIoU:")
    print("-" * 50)
    print(f"{'Class':25} {'mIoU':10} {'Instances':10}")
    print("-" * 50)
    
    for cls in range(len(VOC_CLASSES)):
        if len(class_ious[cls]) > 0:
            miou = np.mean(class_ious[cls])
            instances = len(class_ious[cls])
            print(f"{VOC_CLASSES[cls]:25} {miou:10.4f} {instances:10d}")
        else:
            print(f"{VOC_CLASSES[cls]:25} {'N/A':10} {'0':10}")
    
    final_miou = total_miou / num_samples
    print("-" * 50)
    print(f"Overall mIoU: {final_miou:.4f}")
    
    # Display best and worst examples
    print(f"\nDisplaying best performing example (index {best_idx}, mIoU: {best_miou:.4f})")
    image, target, pred = all_data[best_idx]
    visualize_prediction(image, target, pred, f"Best Performing Example (mIoU: {best_miou:.4f})")
    
    print(f"\nDisplaying worst performing example (index {worst_idx}, mIoU: {worst_miou:.4f})")
    image, target, pred = all_data[worst_idx]
    visualize_prediction(image, target, pred, f"Worst Performing Example (mIoU: {worst_miou:.4f})")
    
    # Visualization loop
    while True:
        idx = input("\nEnter an index to visualize (Q to quit): ")
        if idx.upper() == 'Q':
            break
            
        try:
            idx = int(idx)
            if 0 <= idx < len(all_data):
                image, target, pred = all_data[idx]
                miou = all_mious[idx]
                visualize_prediction(image, target, pred, f"Example {idx} (mIoU: {miou:.4f})")
            else:
                print(f"Index must be between 0 and {len(all_data)-1}")
        except ValueError:
            print("Please enter a valid number or 'Q' to quit")

    return final_miou

if __name__ == "__main__":
    main()
