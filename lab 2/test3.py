import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from model import convNN
from CustomImageDataset import CustomImageDataset
import statistics
import time


def euclidean_distance(pred, true):
    return np.linalg.norm(pred - true)


def calculate_localization_statistics(model, data_loader, device):
    model.eval()
    torch.set_grad_enabled(False)

    distances = []
    min_img, max_img = None, None
    min_dist, max_dist = float('inf'), float('-inf')
    min_pred, max_pred = None, None
    min_label, max_label = None, None

    for imgs, labels in data_loader:
        imgs = imgs.type(torch.float32).to(device=device)
        labels = labels.view(imgs.shape[0], -1).type(torch.float32).to(device=device)

        outputs = model(imgs)
        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        for i in range(len(imgs)):
            dist = euclidean_distance(outputs[i], labels[i])
            distances.append(dist)

            # Track minimum and maximum distances
            if dist < min_dist:
                min_dist = dist
                min_img = imgs[i].cpu()
                min_pred = outputs[i]
                min_label = labels[i]
            if dist > max_dist:
                max_dist = dist
                max_img = imgs[i].cpu()
                max_pred = outputs[i]
                max_label = labels[i]

    mean_dist = statistics.mean(distances)
    std_dev_dist = statistics.stdev(distances)

    return min_dist, max_dist, mean_dist, std_dev_dist, min_img, max_img, min_pred, max_pred, min_label, max_label


def show_images_with_errors(min_img, max_img, min_pred, max_pred, min_label, max_label, min_dist, max_dist):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the image with the lowest error
    axs[0].imshow(min_img.permute(1, 2, 0).numpy())
    axs[0].plot(min_label[0], min_label[1], 'gx', markersize=10, label="True Label")
    axs[0].plot(min_pred[0], min_pred[1], 'rx', markersize=10, label="Predicted Label")
    axs[0].set_title(f"Lowest Error\nDistance: {min_dist:.4f}")
    axs[0].legend()
    axs[0].axis('off')

    # Plot the image with the highest error
    axs[1].imshow(max_img.permute(1, 2, 0).numpy())
    axs[1].plot(max_label[0], max_label[1], 'gx', markersize=10, label="True Label")
    axs[1].plot(max_pred[0], max_pred[1], 'rx', markersize=10, label="Predicted Label")
    axs[1].set_title(f"Highest Error\nDistance: {max_dist:.4f}")
    axs[1].legend()
    axs[1].axis('off')

    plt.show()


def main():
    print('running main ...')

    #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-f', metavar='folder', type=str, required=True, help='path to image folder')
    
    args = argParser.parse_args()

    save_file = "weights32fnr_normval_a5500v2.pth"
    image_folder = "./images"
    
    if args.s != None:
        save_file = args.s
    if args.f != None:
        image_folder  = args.f
    
    print(image_folder)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    print('\t\tusing device ', device)

    train_transform = transforms.Compose([
        transforms.Resize((227, 227))
    ])
    test_transform = train_transform

    test_set = CustomImageDataset("test_noses.txt", image_folder, test_transform, None)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    model = convNN()
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    test_start_time = time.time()
    min_dist, max_dist, mean_dist, std_dev_dist, min_img, max_img, min_pred, max_pred, min_label, max_label = calculate_localization_statistics(
        model, test_loader, device)
    test_end_time = time.time()
    total_test_time = test_end_time - test_start_time
    print("\nfor file: ", save_file)
    print("\ntotal test time: ", total_test_time)
    print(f"Min Distance: {min_dist}")
    print(f"Max Distance: {max_dist}")
    print(f"Mean Distance: {mean_dist}")
    print(f"Standard Deviation: {std_dev_dist}")

    # Show images with lowest and highest errors
    show_images_with_errors(min_img, max_img, min_pred, max_pred, min_label, max_label, min_dist, max_dist)


if __name__ == '__main__':
    main()
