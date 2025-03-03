import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import torch.backends.cudnn as cudnn

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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pre-trained model
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
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
        root='./data',
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
            output = model(images)['out']
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
