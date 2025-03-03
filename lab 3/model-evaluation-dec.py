import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import argparse
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def get_model(model_type):
    """Initialize model architecture matching the training script."""
    if model_type.lower() == "alexnet":
        model = models.alexnet(pretrained=True)
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Replace and unfreeze the classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
        
    elif model_type.lower() == "vgg16":
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
        
    elif model_type.lower() == "resnet18":
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 100)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    total = 0
    top1_correct = 0
    top5_correct = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.type(torch.float32).to(device)  # Match training code's dtype
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            top1_correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_predicted = outputs.topk(5, 1, True, True)
            top5_predicted = top5_predicted.t()
            correct = top5_predicted.eq(labels.view(1, -1).expand_as(top5_predicted))
            top5_correct += correct.any(dim=0).sum().item()
    
    # Calculate all metrics as decimals (between 0 and 1)
    top1_error = 1 - (top1_correct / total)
    top5_error = 1 - (top5_correct / total)
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    
    return {
        'top1_error': top1_error,
        'top5_error': top5_error,
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy
    }

def get_model_type_from_weights(weights_path):
    """Determine model type by looking for model type in the weights filename."""
    filename = weights_path.name.lower()
    if 'alexnet' in filename:
        return 'alexnet'
    elif 'vgg16' in filename:
        return 'vgg16'
    elif 'resnet18' in filename:
        return 'resnet18'
    else:
        # Default to resnet18 as per training script
        return 'resnet18'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', type=str, required=True, help='Directory containing model weight files')
    parser.add_argument('--output_file', type=str, default='evaluation_results.csv', 
                        help='Path to save evaluation results CSV')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for evaluation (default: 64)')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data preprocessing - exactly matching training script
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Load test dataset
    test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Get all .pth files in directory
    weight_files = list(Path(args.weights_dir).glob('*.pth'))
    if not weight_files:
        print(f"No .pth files found in {args.weights_dir}")
        return

    results = []
    
    for weight_file in sorted(weight_files):
        try:
            print(f"\nEvaluating {weight_file.name}...")
            
            # Determine model type and epoch
            model_type = get_model_type_from_weights(weight_file)
            epoch = '5' if weight_file.name.startswith('5_') else 'final'
            
            # Initialize model matching training configuration
            model = get_model(model_type)
            model.to(device)
            
            # Load weights
            state_dict = torch.load(str(weight_file), map_location=device)
            model.load_state_dict(state_dict)
            
            # Evaluate model
            metrics = evaluate_model(model, test_loader, device)
            
            # Store results
            results.append({
                'filename': weight_file.name,
                'model_type': model_type,
                'epoch': epoch,
                'top1_error': metrics['top1_error'],
                'top5_error': metrics['top5_error'],
                'top1_accuracy': metrics['top1_accuracy'],
                'top5_accuracy': metrics['top5_accuracy'],
                'evaluation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Print current results
            print(f"Results for {weight_file.name}:")
            print(f"Model Type: {model_type}")
            print(f"Epoch: {epoch}")
            print(f"Top-1 Error Rate: {metrics['top1_error']:.4f}")
            print(f"Top-5 Error Rate: {metrics['top5_error']:.4f}")
            print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
            print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error processing {weight_file.name}: {str(e)}")
            continue

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to {args.output_file}")
        
        # Print summary of best models
        print("\nBest models by Top-1 Accuracy:")
        best_models = df.nlargest(3, 'top1_accuracy')[['filename', 'model_type', 'epoch', 'top1_accuracy', 'top5_accuracy']]
        print(best_models.to_string(index=False))

if __name__ == '__main__':
    main()
