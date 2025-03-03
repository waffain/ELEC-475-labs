import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import numpy as np
from tqdm import tqdm
import time
import argparse
import torch.backends.cudnn as cudnn

class LightweightDeepLabV3(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        
        # Load the pretrained model
        base_model = deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        
        # Get the backbone
        self.backbone = base_model.backbone
        
        # Create a lighter classifier
        self.classifier = nn.Sequential(
            # Reduce input channels from backbone
            nn.Conv2d(960, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Simplified ASPP
            nn.Conv2d(256, 256, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Dropout for regularization
            nn.Dropout(0.1),
            
            # Final classification layer
            nn.Conv2d(256, num_classes, 1)
        )
        
        # Initialize weights for the new layers
        self._init_weight()

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # Get features from backbone
        features = self.backbone(x)['out']
        
        # Apply classifier
        x = self.classifier(features)
        
        # Upsample to input resolution
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x
    
    def _init_weight(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate LightweightDeepLabV3 model')
    parser.add_argument('--weights', type=str, required=True,
                      help='Path to the trained model weights (weights.pth)')
    parser.add_argument('--data-dir', type=str, default='./data',
                      help='Path to the dataset directory')
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint file."""
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # If the checkpoint contains a full training state
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"Validation mIoU at checkpoint: {checkpoint.get('val_miou', 'unknown')}")
        # If the checkpoint is just the model state dict
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
            
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the LightweightDeepLabV3 model
    model = LightweightDeepLabV3(num_classes=21, pretrained=False)
    
    # Load trained weights
    if not load_checkpoint(model, args.weights, device):
        return

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
        root=args.data_dir,
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

    # Evaluation variables
    total_miou = 0
    num_samples = 0
    total_inference_time = 0
    inference_times = []

    print("Starting evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            targets = targets.to(device)

            # Time the inference
            start_time = time.perf_counter()
            output = model(images)
            end_time = time.perf_counter()
            
            inference_time = end_time - start_time
            total_inference_time += inference_time
            inference_times.append(inference_time)

            # Calculate mIoU
            miou = calculate_miou(output, targets)
            if not np.isnan(miou):
                total_miou += miou
                num_samples += 1

    # Calculate final metrics
    final_miou = total_miou / num_samples
    avg_inference_time = total_inference_time / len(inference_times)
    std_inference_time = np.std(inference_times)

    # Print results
    print(f"\nPerformance Metrics:")
    print(f"Final mIoU: {final_miou:.4f}")
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Average inference time per image: {avg_inference_time*1000:.2f} ms")
    print(f"Standard deviation of inference time: {std_inference_time*1000:.2f} ms")
    print(f"FPS: {1/avg_inference_time:.2f}")

    return final_miou, avg_inference_time


if __name__ == "__main__":
    cudnn.benchmark = False
        
        # Force deterministic algorithms (usually slower)
    cudnn.deterministic = True
        
        # Disable cudnn completely
    cudnn.enabled = False
    main()