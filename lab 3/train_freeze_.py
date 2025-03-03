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
import torchvision.models as models
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import time

# Default parameters
save_file = 'weights_test.pth'
n_epochs = 50
batch_size = 64
plot_file = 'plot_test.png'
image_folder = "./images"
model_type = "alexnet"

def train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, save_file=None, plot_file=None):
    print('Training ...')
    losses_train = []
    losses_val = []

    for epoch in range(1, n_epochs + 1):
        print('Epoch', epoch)

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
                imgs = imgs.type(torch.float32).to(device)
                labels = labels.to(device)

                if phase == 'train':
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    loss_train += loss.item()
                else:
                    with torch.no_grad():
                        outputs = model(imgs)
                        loss = loss_fn(outputs, labels)
                        loss_val += loss.item()

        scheduler.step(loss_train / len(train_loader))

        losses_train.append(loss_train / len(train_loader))
        losses_val.append(loss_val / len(test_loader))

        train_stop_time = time.time()
        print("\nTime to train this epoch: {:.2f} seconds".format(train_stop_time - train_start_time))
        print('{} Epoch {}, Training loss: {:.4f}, Validation loss: {:.4f}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader), loss_val / len(test_loader)))

        if save_file is not None:
            if epoch <= 5:
                torch.save(model.state_dict(), '5_'+save_file)
            else:
                torch.save(model.state_dict(), save_file)

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

def get_model(model_type):
    if model_type.lower() == "alexnet":
        print("Training with AlexNet")
        model = models.alexnet(pretrained=True)
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Replace and unfreeze the classifier
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
        
    elif model_type.lower() == "vgg16":
        print("Training with VGG16")
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
        
    elif model_type.lower() == "resnet18":
        print("Training with ResNet18")
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 100)
        
    else:
        print("No arguments added, training with ResNet18")
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 100)
    
    return model

def main():
    global save_file, n_epochs, batch_size, image_folder, plot_file, model_type
    print('Running main ...')

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='parameter file (.pth)')
    argParser.add_argument('-f', metavar='folder', type=str, help='path to image folder')
    argParser.add_argument('-p', metavar='plot', type=str, help='path to plot')
    argParser.add_argument('-m', metavar='augmentation', type=str, help='augmentation')
    
    args = argParser.parse_args()
    
    if args.s is not None:
        save_file = args.s
    if args.f is not None:
        image_folder = args.f
    if args.p is not None:
        plot_file = args.p
    if args.m is not None:
        model_type = args.m

    print('\t\tn epochs = ', n_epochs)
    print('\t\tbatch size = ', batch_size)
    print('\t\tsave file = ', save_file)
    print('\t\tplot file = ', plot_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t\tusing device ', device)

    model = get_model(model_type)
    model.to(device)

    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=1e-4, weight_decay=1e-5)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])
    
    train_set = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    test_set = CIFAR100(root='./data', train=False, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)
    loss_fn = nn.CrossEntropyLoss()

    train_start_time = time.time()
    print("\nStarting training at: ", train_start_time)
    
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
        plot_file=plot_file
    )
    
    train_end_time = time.time()
    print("\nFor file: ", save_file)
    print("\nEnding training at: ", train_end_time)
    total_train_time = train_end_time - train_start_time
    avg_train_time = total_train_time/n_epochs
    print("\nTotal train time seconds: ", total_train_time)
    print("\nTotal train time minutes: ", total_train_time/60)
    print("\nAverage train time per epoch for ", n_epochs, " epochs ", avg_train_time)

if __name__ == '__main__':
    main()
