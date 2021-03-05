# Bread and butter Mnist CNN just to get acquainted with Pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt

from pathlib import Path

import pdb

class PytorchMnistCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4*4*12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4*4*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def gen_loader(path):
    image_size = 28
    batch_size = 128
    workers = 8
    dataset = dsets.ImageFolder(root=path,
                transform=transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    # transforms.Normalize(tuple(0.1307 for _ in range(nc)), tuple(0.3081 for _ in range(nc))),
                ]))
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=True, num_workers=workers)

def main():
    train_dataroot = Path(__file__).parents[0] / 'data/mnist_png/images/training/'
    test_dataroot = Path(__file__).parents[0] / 'data/mnist_png/images/testing/'
    ngpu = 1
    NUM_EPOCHS = 15

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # device = 'cpu'

    print('creating training dataloader...')
    train_loader = gen_loader(train_dataroot)
    print('finished creating training dataloader')
    
    PRINT_SAMPLE_BATCH = False
    if PRINT_SAMPLE_BATCH:
        samp_batch, labels = next(iter(dataloader))
        npimg = samp_batch[0].numpy()
        grid = vutils.make_grid(samp_batch[:64], nrow=8, padding=2)
        npimg = grid.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
        plt.draw()
        plt.show()

    print('creating network...')
    # pdb.set_trace()
    net = PytorchMnistCnn().to(device)
    print('finished creating network...')

    # Training
    print('starting training loop...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        print(f'starting epoch {epoch}')
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Test the network
    print('creating test loader...')
    test_loader = gen_loader(test_dataroot)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))

    # Save the network
    SAVE_PATH = './PytorchCnn.pth'
    torch.save(net.state_dict(), SAVE_PATH)

if __name__=='__main__':
    main()