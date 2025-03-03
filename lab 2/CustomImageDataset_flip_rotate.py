
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class CustomImageDataset_flip_rotate(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)  # Always flip

        self.rotate_transform = transforms.RandomRotation(degrees=(-15, -15))  # Rotate 15 degrees clockwise

    def __len__(self):
        return 3 * len(self.img_labels)

    def rotate_label(self, label, img_width, img_height, angle_deg):
        """ Rotate the label coordinates by a given angle. """
        angle_rad = torch.tensor(angle_deg * 3.14159 / 180)
        # Convert the label's center point to image coordinates
        x_centered = label[0] - img_width / 2
        y_centered = label[1] - img_height / 2
        # Apply rotation matrix
        x_rot = x_centered * torch.cos(angle_rad) + y_centered * torch.sin(angle_rad)
        y_rot = -x_centered * torch.sin(angle_rad) + y_centered * torch.cos(angle_rad)
        # Return new coordinates, re-centered
        label[0] = x_rot + img_width / 2
        label[1] = y_rot + img_height / 2
        return label

    def __getitem__(self, idx):
        # Determine the type of augmentation (0 = original, 1 = flipped, 2 = noisy)
        aug_type = idx // len(self.img_labels)
        idx = idx % len(self.img_labels)  # Get the actual image index

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)/255


        #get label
        label=[]
        labels_str = self.img_labels.iloc[idx, 1][1:-1].split(',')
        label.append(int(labels_str[0]))
        label.append(int(labels_str[1]))
        #scale label
        label[0]=227*label[0]/image.shape[2]
        label[1]=227*label[1]/image.shape[1]
        label = torch.tensor(label)


        #some image as 4 channel and some have 1 channel
        if image.shape[0]==4:
            image = image [0:3,:,:]
        if image.shape[0]==1:
            image = image.repeat(3,1,1)

        #turn image to 227x227
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)


        # Apply augmentation based on the `aug_type`
        if aug_type == 1:  # Flip the image
            image = self.flip_transform(image)
            label[0] = 227 - label[0]  # Adjust label after horizontal flip
        elif aug_type == 2: # Rotate image 15 degrees clockwise
            image = self.rotate_transform(image)
            label = self.rotate_label(label, img_width=227, img_height=227, angle_deg=-15)  # Adjust label for rotation

        return image, label

#reality check dataset
# image_set = CustomImageDataset_augmented("train_noses.txt","./images",transforms.Compose([transforms.Resize((227,227))]),None)
# print(f"Dataset size: {len(image_set)}")
# train_dataloader = DataLoader(image_set, batch_size=64, shuffle=False)
# # Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].permute(1,2,0)
# label = train_labels[0]
# plt.imshow(img)
# x, y = label[0], label[1]  # Convert tensor values to integers
# print(f"Label: {label}")
# plt.scatter([x], [y], color='red', marker='x')  # Plot the label as a red 'x' mark
# plt.show()

# Reality check dataset
# image_set = CustomImageDataset_flip_rotate("train_noses.txt", "./images", transforms.Compose([transforms.Resize((227, 227))]), None)
# print(f"Dataset size: {len(image_set)}")
#
# # Display image and label for the selected index
# train_dataloader = DataLoader(image_set, batch_size=len(image_set), shuffle=False)
# train_features, train_labels = next(iter(train_dataloader))
#
# while True:
#     # Prompt user for index
#     print(len(train_features))
#     index = int(input("Enter an index: "))
#
#     if index >= 0 and index < len(train_features):
#         img = train_features[index].permute(1, 2, 0)
#         label = train_labels[index]
#
#         plt.imshow(img)
#
#         x, y = label[0].item(), label[1].item()  # Convert tensor values to integers
#         print(f"Label: {label}")
#         plt.scatter([x], [y], color='red', marker='x')  # Plot the label as a red 'x' mark
#         plt.show()
#     else:
#         print(f"Invalid index. Please enter an index between 0 and {len(train_features) - 1}.")