
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import datetime
import argparse
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchsummary import summary
from model import convNN
from CustomImageDataset import CustomImageDataset
from CustomImageDataset_flip import CustomImageDataset_flip
from CustomImageDataset_noise import CustomImageDataset_noise
from CustomImageDataset_augmented import CustomImageDataset_augmented
from CustomImageDataset_rotate import CustomImageDataset_rotate
from CustomImageDataset_flip_noise import CustomImageDataset_flip_noise
from CustomImageDataset_flip_rotate import CustomImageDataset_flip_rotate
from CustomImageDataset_noise_rotate import CustomImageDataset_noise_rotate
import time


#   some default parameters, which can be overwritten by command line arguments
save_file = 'weights_test.pth'
n_epochs = 100
batch_size = 32
plot_file = 'plot_test.png'
image_folder = "./images"
aug_type = "u"


def train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, save_file=None, plot_file=None):
    print('Training ...')
    losses_train = []
    losses_val = []

    for epoch in range(1, n_epochs + 1):
        print('Epoch', epoch)

        # Initialize loss accumulators
        loss_train = 0.0
        loss_val = 0.0

        train_start_time = time.time()

        for phase, loader in [('train', train_loader), ('val', test_loader)]:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for data in loader:
                imgs, labels = data

                # Move inputs and labels to device (GPU or CPU)
                imgs = imgs.type(torch.float32).to(device)
                labels = labels.view(imgs.shape[0], -1)
                labels = labels.type(torch.float32).to(device)

                if phase == 'train':
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    # Accumulate the training loss
                    loss_train += loss.item()

                else:  # Validation phase
                    with torch.no_grad():
                        outputs = model(imgs)
                        loss = loss_fn(outputs, labels)

                        # Accumulate the validation loss
                        loss_val += loss.item()

        # Step the scheduler (depending on if you want to step on training loss or validation loss)
        scheduler.step(loss_train / len(train_loader))

        # Store average losses
        losses_train.append(loss_train / len(train_loader))
        losses_val.append(loss_val / len(test_loader))

        train_stop_time = time.time()
        print("\nTime to train this epoch: {:.2f} seconds".format(train_stop_time - train_start_time))

        # Print epoch summary
        print('{} Epoch {}, Training loss: {:.4f}, Validation loss: {:.4f}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader), loss_val / len(test_loader)))

        # Save the model
        if save_file is not None:
            torch.save(model.state_dict(), save_file)

        # Plot the loss curves
        if plot_file is not None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.plot(losses_val, label='val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            print('Saving plot to', plot_file)
            plt.savefig(plot_file)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():

    global  save_file, n_epochs, batch_size, image_folder, plot_file, aug_type
    print('running main ...')
        #   read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-f', metavar='folder', type=str, required=True, help='path to image folder')
    argParser.add_argument('-p', metavar='plot', type=str, help='path to plot')
    argParser.add_argument('-a', metavar='augmentation', type=str, help='augmentation')
    
    args = argParser.parse_args()
    
    if args.s != None:
        save_file = args.s
    if args.f != None:
        image_folder  = args.f
    if args.p != None:
        plot_file  = args.p
    if args.a != None:
        aug_type  = args.a
        

    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    N_input = 227 * 227   # input image size
    N_output = N_input
    model = convNN()
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)
    print(model.input_shape)


    train_transform = transforms.Compose([
        transforms.Resize((227,227))
    ])
    test_transform = train_transform



    if aug_type == "f":
        print("training with flip augmentation")
        train_set = CustomImageDataset_flip("train_noses.txt", image_folder, train_transform, None)
    elif aug_type == "n":
        print("training with noise augmentation")
        train_set = CustomImageDataset_noise("train_noses.txt", image_folder, train_transform, None)
    elif aug_type == "r":    
        print("training with rotate augmentation")
        train_set = CustomImageDataset_rotate("train_noses.txt", image_folder, train_transform, None)
    elif aug_type == "fn":    
        print("training with flip and noise augmentation")
        train_set = CustomImageDataset_flip_noise("train_noses.txt", image_folder, train_transform, None)
    elif aug_type == "fr":    
        print("training with flip and rotate augmentation")
        train_set = CustomImageDataset_flip_rotate("train_noses.txt",image_folder, train_transform, None)
    elif aug_type == "nr":    
        print("training with noise and rotate augmentation")
        train_set = CustomImageDataset_noise_rotate("train_noses.txt", image_folder, train_transform, None)
    elif aug_type == "fnr":    
        print("training with flip, noise, and rotate augmentation")
        train_set = CustomImageDataset_augmented("train_noses.txt", image_folder, train_transform, None)
    else:    
        print("training with no augmentation")
        train_set = CustomImageDataset("train_noses.txt",image_folder,train_transform,None)

    test_set = CustomImageDataset("test_noses.txt",image_folder,train_transform,None)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    train_start_time=time.time()
    print ("\nstarting training at: ", train_start_time)
    train(
            n_epochs=n_epochs,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            test_loader=test_loader,
            scheduler=scheduler,
            device=device,
            save_file=save_file,
            plot_file = plot_file)
    train_end_time = time.time()
    print("\nfor file: ", save_file)
    print("\nending training at: ", train_end_time)
    total_train_time = train_end_time-train_start_time
    avg_train_time= total_train_time/n_epochs
    print("\ntotal train time seconds: ", total_train_time)
    print("\ntotal train time minutes: ", total_train_time/60)
    print("\naverage train time per epoch for ",n_epochs," epochs ", avg_train_time)
###################################################################

if __name__ == '__main__':
    main()


