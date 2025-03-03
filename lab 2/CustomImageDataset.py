
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



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = read_image(img_path)
        image = read_image(img_path)/255 #scales it between 0 and 1

        #get label
        label=[]
        labels_str = self.img_labels.iloc[idx, 1][1:-1].split(',')
        label.append(int(labels_str[0]))
        label.append(int(labels_str[1]))
        #scale label
        label[0]=227*label[0]/image.shape[2]
        label[1]=227*label[1]/image.shape[1]
        label = torch.tensor(label)
        #label = self.img_labels.iloc[idx, 1]

        #some image as 4 channel and some have 1 channel
        if image.shape[0]==4:
            image = image [0:3,:,:]
        if image.shape[0]==1:
            image = image.repeat(3,1,1)


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# #reality check dataset
# image_set = CustomImageDataset("train_noses.txt","./images",transforms.Compose([transforms.Resize((227,227))]),None)
# print(f"Dataset size: {len(image_set)}")
# train_dataloader = DataLoader(image_set, batch_size=64, shuffle=True)
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
# image_set = CustomImageDataset("train_noses.txt", "./images", transforms.Compose([transforms.Resize((227, 227))]), None)
# print(f"Dataset size: {len(image_set)}")
#
# # Display image and label for the selected index
# train_dataloader = DataLoader(image_set, batch_size=64, shuffle=False)
# train_features, train_labels = next(iter(train_dataloader))
#
# # Prompt user for index
# print(len(train_features))
# index = int(input("Enter an index: "))
#
# if index >= 0 and index < len(train_features):
#     img = train_features[index].permute(1, 2, 0)
#     label = train_labels[index]
#
#     plt.imshow(img)
#
#     x, y = label[0].item(), label[1].item()  # Convert tensor values to integers
#     print(f"Label: {label}")
#     plt.scatter([x], [y], color='red', marker='x')  # Plot the label as a red 'x' mark
#     plt.show()
# else:
#     print(f"Invalid index. Please enter an index between 0 and {len(train_features) - 1}.")
