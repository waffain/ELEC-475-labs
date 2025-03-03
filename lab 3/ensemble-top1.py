import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import torch.nn.functional as F

def get_model(model_type):
    """Initialize model architecture matching the training script."""
    if model_type.lower() == "alexnet":
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
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
    
    return model

def load_models(device, alexnet_path, resnet18_path, vgg16_path):
    """Load all three models and their weights."""
    models_dict = {}
    
    # Load AlexNet
    alexnet = get_model('alexnet')
    alexnet.load_state_dict(torch.load(alexnet_path, map_location=device))
    alexnet.to(device)
    alexnet.eval()
    models_dict['alexnet'] = alexnet
    
    # Load ResNet18
    resnet18 = get_model('resnet18')
    resnet18.load_state_dict(torch.load(resnet18_path, map_location=device))
    resnet18.to(device)
    resnet18.eval()
    models_dict['resnet18'] = resnet18
    
    # Load VGG16
    vgg16 = get_model('vgg16')
    vgg16.load_state_dict(torch.load(vgg16_path, map_location=device))
    vgg16.to(device)
    vgg16.eval()
    models_dict['vgg16'] = vgg16
    
    return models_dict

def get_model_predictions(models_dict, images, device):
    """Get softmax predictions from all models."""
    predictions = {}
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            outputs = model(images)
            predictions[model_name] = F.softmax(outputs, dim=1)
    
    return predictions

def max_probability_ensemble(predictions):
    """Ensemble method using maximum probability."""
    # Stack all predictions and get max probability along model dimension
    all_preds = torch.stack([pred for pred in predictions.values()])
    max_probs, _ = torch.max(all_preds, dim=0)
    return max_probs

def probability_averaging_ensemble(predictions):
    """Ensemble method using average probability."""
    # Stack all predictions and average along model dimension
    all_preds = torch.stack([pred for pred in predictions.values()])
    avg_probs = torch.mean(all_preds, dim=0)
    return avg_probs

def majority_voting_ensemble(predictions):
    """Ensemble method using majority voting."""
    # Get top prediction from each model
    votes = []
    for pred in predictions.values():
        _, vote = torch.max(pred, dim=1)
        votes.append(vote)
    
    # Stack votes and get mode (majority)
    votes = torch.stack(votes)
    majority, _ = torch.mode(votes, dim=0)
    
    # Convert to one-hot encoding
    device = votes.device
    one_hot = torch.zeros(majority.size(0), 100, device=device)
    one_hot.scatter_(1, majority.unsqueeze(1), 1)
    return one_hot

def evaluate_ensemble(ensemble_method, predictions, labels):
    """Evaluate ensemble predictions and calculate metrics."""
    if ensemble_method == "max_probability":
        final_probs = max_probability_ensemble(predictions)
    elif ensemble_method == "probability_averaging":
        final_probs = probability_averaging_ensemble(predictions)
    else:  # majority_voting
        final_probs = majority_voting_ensemble(predictions)
    
    # Calculate top-1 metric only
    _, top1_pred = final_probs.max(1)
    top1_correct = top1_pred.eq(labels).sum().item()
    
    return top1_correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alexnet_weights', type=str, required=True, help='Path to AlexNet weights')
    parser.add_argument('--resnet18_weights', type=str, required=True, help='Path to ResNet18 weights')
    parser.add_argument('--vgg16_weights', type=str, required=True, help='Path to VGG16 weights')
    parser.add_argument('--output_file', type=str, default='ensemble_results.csv', help='Output file path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load models
    models_dict = load_models(
        device,
        args.alexnet_weights,
        args.resnet18_weights,
        args.vgg16_weights
    )
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # Initialize counters for each ensemble method
    ensemble_methods = ['max_probability', 'probability_averaging', 'majority_voting']
    results = {method: {'top1_correct': 0} for method in ensemble_methods}
    total = 0
    
    # Evaluate
    for images, labels in test_loader:
        images = images.type(torch.float32).to(device)
        labels = labels.to(device)
        total += labels.size(0)
        
        # Get predictions from all models
        predictions = get_model_predictions(models_dict, images, device)
        
        # Evaluate each ensemble method
        for method in ensemble_methods:
            top1_correct = evaluate_ensemble(method, predictions, labels)
            results[method]['top1_correct'] += top1_correct
    
    # Calculate final metrics and save results
    output_data = []
    for method in ensemble_methods:
        top1_accuracy = results[method]['top1_correct'] / total
        top1_error = 1.0 - top1_accuracy
        
        output_data.append({
            'ensemble_method': method,
            'error_rate': top1_error,
            'accuracy': top1_accuracy
        })
        
        print(f"\nResults for {method}:")
        print(f"Error Rate: {top1_error:.4f}")
        print(f"Accuracy: {top1_accuracy:.4f}")
    
    # Save results to CSV
    df = pd.DataFrame(output_data)
    df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")

if __name__ == '__main__':
    main()
